import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator



PROJECT_ROOT = "/home/infres/diouf-25/PRIM-Project"
MODEL_NAME   = "Wnet"

EXP_ROOT = os.path.join(PROJECT_ROOT, "experiment", MODEL_NAME)
LOG_DIR = os.path.join(EXP_ROOT, "logs")

# Dossiers output
FIG_ROOT     = os.path.join(EXP_ROOT, "figures")
FIG_PER_RUN  = os.path.join(FIG_ROOT, "per_run")
FIG_ALL_RUNS = os.path.join(FIG_ROOT, "all_runs")

os.makedirs(FIG_ROOT, exist_ok=True)
os.makedirs(FIG_PER_RUN, exist_ok=True)
os.makedirs(FIG_ALL_RUNS, exist_ok=True)




def load_scalars(run_path):
    """
    Lit toutes les courbes scalaires d’un run TensorBoard
    Retourne : dict[tag] = (steps, values)
    """

    ea = EventAccumulator(run_path)
    ea.Reload()

    scalars = {}

    if "scalars" not in ea.Tags():
        return scalars

    for tag in ea.Tags()["scalars"]:

       
        if tag.startswith("Test/"):
            continue

        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]

        scalars[tag] = (steps, values)

    return scalars



# Export d’une figure par TAG et par RUN

def export_per_run(run_name, scalars):
    for tag, (steps, values) in scalars.items():

        safe_tag = tag.replace("/", "_")
        filename = f"{MODEL_NAME}_{run_name}_{safe_tag}.png"
        filepath = os.path.join(FIG_PER_RUN, filename)

        plt.figure(figsize=(8, 4))
        plt.plot(steps, values)
        plt.title(f"{tag} - {run_name}")
        plt.xlabel("step")
        plt.ylabel(tag)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150)
        plt.close()



# Export d’une figure par TAG contenant TOUS les RUNS

def export_all_runs(all_scalars):

    # liste de tous les tags existants
    all_tags = set()
    for s in all_scalars.values():
        all_tags |= set(s.keys())

    for tag in sorted(all_tags):

        plt.figure(figsize=(8, 4))

        for run_name, scalars in all_scalars.items():

            if tag not in scalars:
                continue

            steps, values = scalars[tag]
            plt.plot(steps, values, label=run_name)

        plt.title(f"{tag} - ALL RUNS")
        plt.xlabel("step")
        plt.ylabel(tag)
        plt.grid(True)
        plt.legend()

        safe_tag = tag.replace("/", "_")
        filename = f"ALLRUNS_{safe_tag}.png"
        filepath = os.path.join(FIG_ALL_RUNS, filename)

        plt.tight_layout()
        plt.savefig(filepath, dpi=150)
        plt.close()



# MAIN

def main():
    print("Lecture des logs dans :", LOG_DIR)

    all_scalars = {}

    # ----- lire tous les runs -----
    for run_name in sorted(os.listdir(LOG_DIR)):

        run_path = os.path.join(LOG_DIR, run_name)
        if not os.path.isdir(run_path):
            continue

        print(f"Chargement {run_name} ...")

        scalars = load_scalars(run_path)
        all_scalars[run_name] = scalars

        export_per_run(run_name, scalars)

    # ----- exporter les figures combinées -----
    export_all_runs(all_scalars)

    print("\n Export terminé !")
    print("Figures individuelles par RUN →", FIG_PER_RUN)
    print("Figures ALL RUNS combinés     →", FIG_ALL_RUNS)


if __name__ == "__main__":
    main()
