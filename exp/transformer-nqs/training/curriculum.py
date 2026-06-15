"""
Curriculum scheduler for progressive difficulty training.

Training proceeds in stages. Each stage introduces a new difficulty tier and
keeps all previous tiers (cumulative curriculum). The model first masters
small, easy circuits before being exposed to harder ones.

Example
-------
    ve = load_dataset(difficulties=("very_easy",))
    ea = load_dataset(difficulties=("easy",))
    mo = load_dataset(difficulties=("moderate",))

    scheduler = CurriculumScheduler([
        ("very_easy", ve, 50),    # epochs 0–49:  only very_easy
        ("easy",      ea, 100),   # epochs 50–149: very_easy + easy
        ("moderate",  mo, 150),   # epochs 150–299: all three tiers
    ])

    for epoch in range(scheduler.total_epochs):
        samples = scheduler.get_samples(epoch)   # cumulative data for this epoch
        loss    = train_epoch(model, opt, samples)
"""


class CurriculumScheduler:
    """
    Parameters
    ----------
    stages : list of (name, samples, n_epochs) tuples
        Each stage adds its samples to all previous ones.
        n_epochs is the number of epochs spent *entering* this stage
        (i.e. the stage is active from the start of its epoch range).
    """

    def __init__(self, stages):
        self._stages  = stages          # [(name, samples, n_epochs), ...]
        self._boundaries = []           # cumulative epoch counts where each stage starts
        cumulative = 0
        for _, _, n in stages:
            self._boundaries.append(cumulative)
            cumulative += n
        self._total = cumulative

    @property
    def total_epochs(self):
        return self._total

    def _stage_index(self, epoch):
        """Return the 0-based index of the active stage at a given epoch."""
        idx = 0
        for i, boundary in enumerate(self._boundaries):
            if epoch >= boundary:
                idx = i
        return idx

    def get_stage(self, epoch):
        """Return the name of the active stage at this epoch."""
        return self._stages[self._stage_index(epoch)][0]

    def get_samples(self, epoch):
        """
        Return all samples available at this epoch (cumulative up to and
        including the current stage).
        """
        active_idx = self._stage_index(epoch)
        combined = []
        for i in range(active_idx + 1):
            combined.extend(self._stages[i][1])
        return combined
