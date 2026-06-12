from abc import ABC, abstractmethod

class ModelRunner(ABC):
    @abstractmethod
    def build_cmd(self, config) -> str:
        """Return the shell command string to run inference."""
        ...

    @abstractmethod
    def prepare(self, config) -> dict:
        """Do any pre-run setup (e.g. write config files). Return a dict of
        anything _execute needs (e.g. {'configFile': path})."""
        ...

    @abstractmethod
    def validate(self, config) -> str | None:
           if not config["simulation_name"].strip():
               return "Error: Please enter a simulation name."
           return None
