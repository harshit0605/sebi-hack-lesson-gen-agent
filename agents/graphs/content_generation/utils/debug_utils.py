from agents.graphs.content_generation.state import LessonCreationState


def print_without_pdf_content(state: LessonCreationState):
    """Print state without pdf_content"""
    filtered = {k: v for k, v in state.items() if k != "pdf_content"}
    print(filtered)
