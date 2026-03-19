"""
RAG Evaluation System
Fixes applied:
  1. _calculate_confidence  — text answers now score 0.65–0.75 without SQL/data
  2. _detect_hallucination  — removed 'typically/usually/generally' false-positives;
                              kept only unambiguous AI-hedge phrases
  3. _determine_quality     — only averages metrics that were actually computed
"""

import re
import sys
from pathlib import Path
from typing import Dict

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

CONFIG_PATH = ROOT_DIR / 'config' / 'config.yaml'


class RAGEvaluator:
    """Evaluate RAG responses for quality and accuracy"""

    def __init__(self, config_path: str = None):
        self.eval_config = {
            'faithfulness':      {'enabled': True, 'threshold': 0.7},
            'hallucination':     {'enabled': True, 'threshold': 0.2},   # tightened
            'context_relevance': {'enabled': True, 'min_score': 0.5},
            'confidence':        {'enabled': True, 'min_score': 0.6}
        }

        if config_path is None:
            config_path = CONFIG_PATH

        try:
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)
            cfg = config.get('evaluation', {})
            for key in self.eval_config:
                if key in cfg:
                    self.eval_config[key].update(cfg[key])
            # Always keep hallucination threshold at 0.2
            self.eval_config['hallucination']['threshold'] = 0.2
        except Exception as e:
            print(f"⚠️  Using default evaluation config: {e}")

        print("✅ RAG Evaluator initialized")

    # ─────────────────────────────────────────────
    # Public
    # ─────────────────────────────────────────────

    def evaluate_response(
        self,
        question: str,
        answer:   str,
        context:  str = "",
        sql:      str = "",
        data:     str = ""
    ) -> Dict:
        """Comprehensive evaluation of a RAG response."""

        evaluation = {
            'faithfulness_score':     0.0,
            'hallucination_detected': False,
            'hallucination_score':    0.0,
            'context_relevance':      0.0,
            'confidence_score':       0.0,
            'overall_quality':        'unknown',
            'flags':                  [],
            'passed':                 True
        }

        # 1. Faithfulness (only when context is present)
        if context:
            evaluation['faithfulness_score'] = self._calculate_faithfulness(
                answer, context
            )
            if (evaluation['faithfulness_score']
                    < self.eval_config['faithfulness']['threshold']):
                evaluation['flags'].append('low_faithfulness')

        # 2. Hallucination detection
        hall = self._detect_hallucination(question, answer, context, data)
        evaluation['hallucination_detected'] = hall['detected']
        evaluation['hallucination_score']    = hall['score']
        if hall['detected']:
            evaluation['flags'].append('hallucination_detected')
            evaluation['passed'] = False

        # 3. Context relevance (only when context is present)
        if context:
            evaluation['context_relevance'] = self._calculate_context_relevance(
                question, context
            )
            if (evaluation['context_relevance']
                    < self.eval_config['context_relevance']['min_score']):
                evaluation['flags'].append('low_context_relevance')

        # 4. Confidence
        evaluation['confidence_score'] = self._calculate_confidence(
            question, answer, sql, data
        )
        if (evaluation['confidence_score']
                < self.eval_config['confidence']['min_score']):
            evaluation['flags'].append('low_confidence')

        # 5. Overall quality
        evaluation['overall_quality'] = self._determine_quality(evaluation)

        return evaluation

    # ─────────────────────────────────────────────
    # Faithfulness
    # ─────────────────────────────────────────────

    def _calculate_faithfulness(self, answer: str, context: str) -> float:
        if not answer or not context:
            return 0.0
        a_words = set(re.findall(r'\w+', answer.lower()))
        c_words = set(re.findall(r'\w+', context.lower()))
        if not a_words:
            return 0.0
        return min(len(a_words & c_words) / len(a_words), 1.0)

    # ─────────────────────────────────────────────
    # Hallucination  (FIXED — no false positives)
    # ─────────────────────────────────────────────

    # Only phrases that unambiguously signal the model is guessing /
    # drawing on training rather than retrieved data.
    # REMOVED: 'typically', 'usually', 'generally' — perfectly normal English.
    _HALLUCINATION_PATTERNS = [
        r'based on my (knowledge|training|experience)',
        r'\bi (know|believe|think) that\b',
        r'in my (experience|opinion)',
        r'as (an AI|a language model)',
        r"i (cannot|can't) (verify|confirm|access)",
        r'my training data (suggests|indicates|shows)',
    ]

    def _detect_hallucination(
        self, question: str, answer: str, context: str, data: str
    ) -> Dict:
        result = {'detected': False, 'score': 0.0, 'reasons': []}

        a_lower = answer.lower()
        for pattern in self._HALLUCINATION_PATTERNS:
            if re.search(pattern, a_lower):
                result['score'] += 0.25
                result['reasons'].append(f"Pattern: '{pattern}'")

        # Fabricated numbers (only when data is available to compare against)
        if data and data not in ('', 'None', '[]', '{}'):
            a_nums     = set(re.findall(r'\b\d+\.?\d*\b', answer))
            d_nums     = set(re.findall(r'\b\d+\.?\d*\b', str(data)))
            fabricated = a_nums - d_nums
            if fabricated and len(fabricated) > 2:
                result['score'] += 0.30
                result['reasons'].append(f"Fabricated numbers: {fabricated}")

        result['score'] = min(result['score'], 1.0)

        # threshold = 0.2 so one clear pattern → detected
        if result['score'] >= self.eval_config['hallucination']['threshold']:
            result['detected'] = True

        return result

    # ─────────────────────────────────────────────
    # Context relevance
    # ─────────────────────────────────────────────

    def _calculate_context_relevance(self, question: str, context: str) -> float:
        if not question or not context:
            return 0.0
        q_words = set(re.findall(r'\w+', question.lower()))
        c_words = set(re.findall(r'\w+', context.lower()))
        if not q_words:
            return 0.0
        return min(len(q_words & c_words) / len(q_words), 1.0)

    # ─────────────────────────────────────────────
    # Confidence  (REWRITTEN)
    # ─────────────────────────────────────────────

    def _calculate_confidence(
        self, question: str, answer: str, sql: str, data: str
    ) -> float:
        """
        Points breakdown (max 1.0):
          0.40  non-trivial answer (> 10 chars)
          0.25  word count ≥ 20        (else 0.15 for ≥ 8)
          0.20  valid SQL (SELECT … ;)
          0.15  actual data returned
          0.10  structured / markdown answer

        A well-formed text answer (no SQL, no data) scores 0.65–0.75.
        A SQL answer with data scores up to 1.0.
        """
        score = 0.0

        if answer and len(answer) > 10:
            score += 0.40

        wc = len(answer.split()) if answer else 0
        if wc >= 20:
            score += 0.25
        elif wc >= 5:       # short but direct answers (e.g. "There are 8 artists.")
            score += 0.20

        if sql and 'SELECT' in sql.upper() and ';' in sql:
            score += 0.20

        if data and data not in ('', 'None', '[]', '{}'):
            score += 0.15

        if any(m in answer for m in ['**', '- ', '• ', '###', '\n\n', '|']):
            score += 0.10

        return min(score, 1.0)

    # ─────────────────────────────────────────────
    # Quality gate  (FIXED)
    # ─────────────────────────────────────────────

    def _determine_quality(self, evaluation: Dict) -> str:
        """
        Weighted quality scoring:
        - confidence_score carries 70% weight (always present, most reliable)
        - faithfulness_score carries 20% only when context was provided
        - context_relevance carries 10% only when context was provided

        Prevents image/PDF/SQL answers being rated 'poor' just because their
        long extracted text doesn't overlap with a short memory context string.
        """
        if evaluation['hallucination_detected']:
            return 'poor'

        conf    = evaluation['confidence_score']
        faith   = evaluation['faithfulness_score']
        ctx_rel = evaluation['context_relevance']

        has_faith = faith > 0
        has_ctx   = ctx_rel > 0

        if has_faith and has_ctx:
            weighted = conf * 0.70 + faith * 0.20 + ctx_rel * 0.10
        elif has_faith:
            weighted = conf * 0.80 + faith * 0.20
        elif has_ctx:
            weighted = conf * 0.90 + ctx_rel * 0.10
        else:
            weighted = conf   # no context — confidence is sole signal

        if weighted >= 0.75:
            return 'excellent'
        elif weighted >= 0.55:
            return 'good'
        elif weighted >= 0.35:
            return 'fair'
        else:
            return 'poor'

    # ─────────────────────────────────────────────
    # Report
    # ─────────────────────────────────────────────

    def create_eval_report(self, evaluation: Dict) -> str:
        no_ctx  = " (N/A — no context)"
        report  = "📊 EVALUATION REPORT\n"
        report += "=" * 60 + "\n\n"
        report += f"Overall Quality : {evaluation['overall_quality'].upper()}\n"
        report += f"Passed          : {'✅ YES' if evaluation['passed'] else '❌ NO'}\n\n"
        report += "Scores:\n"
        report += f"  • Confidence        : {evaluation['confidence_score']:.2f}\n"
        report += (
            f"  • Faithfulness      : {evaluation['faithfulness_score']:.2f}"
            + (no_ctx if evaluation['faithfulness_score'] == 0 else "") + "\n"
        )
        report += (
            f"  • Context Relevance : {evaluation['context_relevance']:.2f}"
            + (no_ctx if evaluation['context_relevance'] == 0 else "") + "\n"
        )
        report += f"  • Hallucination     : {evaluation['hallucination_score']:.2f}\n\n"
        if evaluation['flags']:
            report += "⚠️  Flags:\n"
            for flag in evaluation['flags']:
                report += f"  • {flag}\n"
        else:
            report += "✅ No flags raised.\n"
        return report


# ─────────────────────────────────────────────
# Smoke tests
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing RAG Evaluator...\n")
    evaluator = RAGEvaluator()

    cases = [
        {
            "name":           "SQL answer — no context",
            "question":       "How many artists?",
            "answer":         "There are 8 artists in the database.",
            "sql":            "SELECT COUNT(*) FROM artists;",
            "data":           "count: 8",
            "expect_quality": ["excellent", "good"],
            "expect_hall":    False,
        },
        {
            "name":     "Rich text answer — no context",
            "question": "Tell me about the database",
            "answer":   (
                "The sales analytics database contains the following tables:\n\n"
                "  - **artists** (8 rows)\n"
                "  - **albums** (347 rows)\n"
                "  - **sales** (2240 rows)\n\n"
                "Switch to the SQL Query tab to run live queries."
            ),
            "expect_quality": ["excellent", "good"],
            "expect_hall":    False,
        },
        {
            "name":     "Answer with context",
            "question": "How many artists?",
            "answer":   "There are 8 artists in the database.",
            "context":  "The database contains 8 artists.",
            "data":     "count: 8",
            "expect_quality": ["excellent", "good"],
            "expect_hall":    False,
        },
        {
            "name":     "Hallucination — AI hedge phrase",
            "question": "How many artists?",
            "answer":   "Based on my knowledge there are around 50 artists.",
            "context":  "The database contains 8 artists.",
            "expect_quality": ["poor"],
            "expect_hall":    True,
        },
        {
            "name":     "'Generally' is NOT a hallucination",
            "question": "What does the database contain?",
            "answer":   "The database generally covers sales transactions.",
            "expect_quality": ["excellent", "good", "fair"],
            "expect_hall":    False,
        },
    ]

    all_ok = True
    for c in cases:
        r = evaluator.evaluate_response(
            question=c.get("question", ""),
            answer=c.get("answer", ""),
            context=c.get("context", ""),
            sql=c.get("sql", ""),
            data=c.get("data", ""),
        )
        q_ok = r['overall_quality'] in c['expect_quality']
        h_ok = r['hallucination_detected'] == c['expect_hall']
        ok   = q_ok and h_ok
        if not ok:
            all_ok = False
        icon = "✅" if ok else "❌"
        print(
            f"{icon} {c['name']}\n"
            f"   quality={r['overall_quality']} (expected {c['expect_quality']})"
            f"  conf={r['confidence_score']:.2f}"
            f"  hall={r['hallucination_detected']} (expected {c['expect_hall']})"
            f"  flags={r['flags']}\n"
        )

    print("=" * 60)
    print("✅ All tests passed!" if all_ok else "❌ Some tests FAILED")
    print()
    print(evaluator.create_eval_report(
        evaluator.evaluate_response(
            question="How many artists?",
            answer="There are 8 artists in the database.",
            sql="SELECT COUNT(*) FROM artists;",
            data="count: 8"
        )
    ))