import openml
from tqdm import tqdm


def download(suite_id):
    suite = openml.study.get_suite(suite_id)
    for task_id in tqdm(suite.tasks):
        task = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()
        dataset.get_data(target=task.target_name)


def main():
    download(353)


if __name__ == "__main__":
    main()
