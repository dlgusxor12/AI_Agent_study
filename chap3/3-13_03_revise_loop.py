def approve(state):
    raw = interrupt({
        "question": "이 초안을 (ok / no / revise: <지시>)",
        "draft": state["draft"],
    })
    text = str(raw).strip()
    if text.lower().startswith("revise"):
        _, _, feedback = text.partition(":")
        return {"decision": "revise", "feedback": feedback.strip()}
    return {"decision": text.lower()}


def after_approve(state):
    if state["decision"] == "revise":
        return "draft_mail"  # 다시 초안으로
    return "send_mail"


builder.add_conditional_edges(
    "approve", after_approve,
    {"draft_mail": "draft_mail", "send_mail": "send_mail"},
)