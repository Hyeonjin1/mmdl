"""Refresh the results section of the MMMU Lookback experiment record."""
import json
from pathlib import Path
import sys


MARKER = "<!-- generated results -->"


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pct(value):
    return f"{100*value:.1f}%" if value is not None else "—"


def stage_table(directory):
    data = json.loads((directory / "summary.json").read_text())
    lines = [
        f"저장된 짝 {data['pairs']}/{data['expected_per_condition']}.",
        "",
        "| 문항군 | 조건 | 완료 | 정확도 | 추출 | 길이 종료 | 반복 종료 | 평균 생성 / 삽입 토큰 | 평균 / 합계 초 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group, stats in data["groups"].items():
        if not stats["expected"]:
            continue
        for condition in ("baseline", "lookback"):
            s = stats[condition]
            means = s["means"]
            tokens = f"{means['generated_tokens']:.1f} / {means['insertion_tokens']:.1f}" if means else "—"
            elapsed = f"{means['elapsed_seconds']:.2f} / {s['totals']['elapsed_seconds']:.2f}" if means else "—"
            lines.append(
                f"| {group} | {condition} | {s['completed']}/{stats['expected']} | "
                f"{s['correct']}/{s['completed']} ({pct(s['accuracy'])}) | "
                f"{s['extracted']}/{s['completed']} ({pct(s['extraction_rate'])}) | "
                f"{s['length_count']} | {s['finish_reasons'].get('repetition_abort', 0)} | {tokens} | {elapsed} |"
            )
    lines.append("")
    if not data["complete"]:
        lines.extend(["위 비율은 저장된 각 조건 응답만 분모로 삼은 중간 집계다. 전환은 두 응답이 모두 있는 ID에서 계산했다.", ""])
    for group, stats in data["groups"].items():
        if not stats["pairs"]:
            continue
        transitions = stats["transitions"]
        s = stats["lookback"]
        lines.append(
            f"- {group} {stats['pairs']}쌍에서 오답→정답 {transitions.get('incorrect->correct', 0)}건, "
            f"미추출→정답 {transitions.get('unextracted->correct', 0)}건, "
            f"정답→오답 {transitions.get('correct->incorrect', 0)}건, "
            f"정답→미추출 {transitions.get('correct->unextracted', 0)}건."
        )
        lines.append(
            f"- {group} Lookback {s['intervened_questions']}문항 개입, 총 {s['totals']['interventions']}회 삽입. "
            f"첫 답 단계 신호 {s['first_answer_phase_seen']}문항, 신호 이후 매칭 {s['blocked_pause_seen']}문항. "
            f"재요청 프롬프트 {s['resubmitted_prompt_tokens']}토큰, 캐시 적중으로 보고된 "
            f"{s['cached_resubmitted_prompt_tokens']}토큰, 재요청 첫 출력까지 합계 "
            f"{s['resubmission_first_output_seconds']:.3f}초."
        )
    return lines


def finish_examples(output):
    pairs = jsonl(output / "paired_results.jsonl")
    lines = []
    transitions = ("unextracted->correct", "incorrect->correct", "correct->incorrect", "correct->unextracted")
    for transition in transitions:
        examples = [r for r in pairs if r["transition"] == transition]
        for r in examples[:3]:
            a, b = r["baseline"], r["lookback"]
            lines.append(
                f"- `{r['question_id']}` ({transition}, 정답 `{r['gold']}`). "
                f"대조군 `{a['prediction']}` / {a['finish_reason']}, "
                f"Lookback `{b['prediction']}` / {b['finish_reason']}, 개입 {b['interventions']}회."
            )
    blocked = [r for r in pairs if r["lookback"]["first_blocked_pause"] and not r["lookback"]["interventions"]]
    for r in blocked[:3]:
        phase = r["lookback"]["first_answer_phase"]
        pause = r["lookback"]["first_blocked_pause"]
        lines.append(
            f"- `{r['question_id']}`의 첫 답 신호는 생성 토큰 {phase['generated_position']}, "
            f"답 단계 이후 불확실성 문구는 {pause['generated_position']}. 개입 0회, "
            f"종료 `{r['lookback']['finish_reason']}`."
        )
    return lines


def update(output):
    path = output / "experiment.md"
    content = path.read_text().split(MARKER, 1)[0].rstrip()
    lines = [content, "", MARKER, "", "## 실행 결과", ""]
    smoke = output / "smoke"
    if (smoke / "summary.json").exists():
        lines.extend(["### Smoke test", "",
                      "통과." if (smoke / "passed.json").exists() else "진행 중 또는 미통과.", "",
                      *stage_table(smoke), ""])
        for r in jsonl(smoke / "lookback/predictions.jsonl"):
            positions = [e["generated_position"] for e in r["events"]]
            phase = r["first_answer_phase"]
            answer = f"{phase['phrase']} @ {phase['generated_position']}" if phase else "미검출"
            lines.append(f"- `{r['question_id']}` 발동 위치 `{positions}`, 첫 답 단계 신호 `{answer}`.")
        lines.append("")
    if (output / "summary.json").exists():
        lines.extend(["### 80문항 파일럿", "", *stage_table(output), ""])
        examples = finish_examples(output)
        if examples:
            lines.extend(["### 답이 바뀐 사례와 조기 차단", "", *examples, ""])
    temporary = path.with_suffix(".tmp")
    temporary.write_text("\n".join(lines).rstrip() + "\n")
    temporary.replace(path)


if __name__ == "__main__":
    update(Path(sys.argv[1]))
