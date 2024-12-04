from cycleqd import CycleQD
from cfg import tasks, config

cycle_qd = CycleQD(config)

begin_perfs = []
for task in tasks:
    begin_perfs.append(task.evaluate(cycle_qd.base_model))

cycle_qd.optimize()
final_model = cycle_qd.final()

for task, be_performance in zip(tasks, begin_perfs):
    performance = task.evaluate(final_model)
    print(
        f"Result Performance / Task: {task.name} / Before: {be_performance.item()} / After: {performance.item()}"
    )

final_model.model.save_pretrained("./results")
