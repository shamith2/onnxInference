# Script to run custom AI Recall pipeline

import os
from pathlib import Path

from onnxInsights.aiRecall import AI_Recall_pipeline

# global variables: use with caution

ROOT = Path(__file__).parents[2].resolve()
WORKSPACE = Path(__file__).parent.resolve()

MODEL_DIR = os.path.join(ROOT, 'weights', 'aiRecall')

SNAPSHOT_DIRECTORY = os.path.join(ROOT, 'results', 'aiRecall', 'snapshots')

RESULT_DIR = os.path.join(ROOT, 'results', 'aiRecall')


# runs

def run_recall():
    AI_Recall_pipeline(
        MODEL_DIR,
        query_or_screenshot='Microsoft Keynote',
        top_p=3,
        save_directory=RESULT_DIR
    )

def capture_screenshots():
    AI_Recall_pipeline(
        MODEL_DIR,
        query_or_screenshot=None
    )


if __name__ == '__main__':
    run_recall()
