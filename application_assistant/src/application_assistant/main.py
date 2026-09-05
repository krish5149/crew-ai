#!/usr/bin/env python
import os
from application_assistant.crews.content_crew.ResumeGenerateCrew import ResumeGenerateCrew as JobApplicationCrew


def run():
    """
    Run the Job Application crew with candidate-specific inputs.
    """
    inputs = {
        "job_title": "Entry-level Python Developer",
        "location": "Remote, India",
        "candidate_profile": (
            "Fresh graduate with a B.Tech in Computer Science. Skilled in "
            "Python, SQL, Django, REST APIs, and Git. Completed 2 internships "
            "building backend services. Looking for entry-level Python "
            "developer roles."
        ),
        "candidate_resume": r"/workspaces/bles/application_assistant/resume.pdf"
    }

    result = JobApplicationCrew().crew().kickoff(inputs=inputs)

    print("\n\n===== CREW RUN COMPLETE =====")
    print(result)


if __name__ == "__main__":
    run()