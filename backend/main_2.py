
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(cfg.prepare.lag)

if __name__ == "__main__":
    main()
