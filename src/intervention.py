def suggest_interventions(student):
    """Simple support suggestions; these are not medical/diagnostic decisions."""
    suggestions = []
    absences = student.get("absences", 0) or 0
    failures = student.get("failures", 0) or 0
    studytime = student.get("studytime", 2)
    first_grade = student.get("G1", 10)

    if absences >= 10:
        suggestions.append("Offer a supportive attendance check-in and ask whether barriers are affecting attendance.")
    if failures >= 1:
        suggestions.append("Offer subject-specific practice and a faculty mentoring meeting.")
    if studytime == 1:
        suggestions.append("Help the student create a realistic weekly study timetable.")
    if first_grade < 10:
        suggestions.append("Offer revision resources, practice questions, and a follow-up check-in.")
    if not suggestions:
        suggestions.append("Continue regular progress monitoring and invite the student to ask for support.")
    return suggestions
