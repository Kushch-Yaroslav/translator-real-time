from dataclasses import dataclass


@dataclass
class SystemAudioDevice:
    pulse_name: str
    description: str
    is_default: bool = False

    def label(self) -> str:
        if self.is_default:
            return f"{self.description} [default]"
        return self.description