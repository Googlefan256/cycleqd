from lib import CycleQD, Archive
from cfg import tasks, base_model, sampler, merger, mutator, steps, storage, step_steps

print("Init archive")
archive = Archive(tasks)
print("Init CycleQD")
cycle_qd = CycleQD(base_model, archive, tasks, sampler, merger, mutator, storage)

for i in range(steps):
    task = tasks[i % len(tasks)]
    print(f"Step: {i + 1} / Task: {task.name}")
    for _ in range(step_steps):
        cycle_qd.step(archive, task.name)
    best, score = cycle_qd.best(archive)
    best.model.save_pretrained("./results")
    with open("./results/score.txt", "w") as w:
        w.write(f"Score: {score}")

best, score = cycle_qd.best(archive)
best.model.save_pretrained("./results")
with open("./results/score.txt", "w") as w:
    w.write(f"Score: {score}")
