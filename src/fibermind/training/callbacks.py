import subprocess
import lightning as L


class RcloneCheckpointBackup(L.Callback):
    """Backs up checkpoints to Google Drive every N epochs. Non-blocking."""
    def __init__(self, local_dir="checkpoints",
                 remote_dir="gdrive:FiberMind/runs", every_n=5):
        self.local_dir = local_dir
        self.remote_dir = remote_dir
        self.every_n = every_n

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.every_n == 0 or epoch == trainer.max_epochs - 1:
            run_name = trainer.logger.version if trainer.logger else "run"
            dest = f"{self.remote_dir}/{run_name}/checkpoints"
            subprocess.Popen(["rclone", "copy", self.local_dir, dest,
                              "--transfers=2"])
