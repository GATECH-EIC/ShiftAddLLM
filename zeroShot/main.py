import json

import evaluator
from utils import parse_args


def main(taskname):
    args = parse_args()

    task_names = [taskname]

    print(f"Selected Tasks: {task_names}")

    results = evaluator.simple_evaluate(
        args=args,
        tasks_list=task_names,
    )

    dumped = json.dumps(results, indent=2)
    print(dumped)

    if args.output_path:
        with open(args.output_path, "w") as f:
            f.write(dumped)

    print(
        f"{args.model}"
        f"num_fewshot: {args.num_fewshot},"
        f" batch_size: {args.batch_size}"
    )
    if args.table_results:
        print(evaluator.make_table(results))
    else:
        from pprint import pprint
        pprint(results)


if __name__ == "__main__":
    task_names = ["arc_challenge", "arc_easy", "copa", "boolq", "piqa", "storycloze", "rte"]
    for each in task_names:
        main(each)
