import csv
from collections import defaultdict

FEEDBACK_FILE = "feedback.csv"


def save_feedback(query, company, feedback):
    with open(FEEDBACK_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([query, company, feedback])


def load_feedback_scores():
    """
    Returns:
    {
        "Alpha Metals": {"positive": 3, "negative": 1},
        "Omega Castings": {"positive": 1, "negative": 0}
    }
    """
    scores = defaultdict(lambda: {"positive": 0, "negative": 0})

    try:
        with open(FEEDBACK_FILE, mode="r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                company = row["company"]
                fb = row["feedback"]
                if fb in scores[company]:
                    scores[company][fb] += 1
    except FileNotFoundError:
        pass

    return scores