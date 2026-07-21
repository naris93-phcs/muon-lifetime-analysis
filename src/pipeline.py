from collections.abc import Iterable

from src.detector import analyze_event
from src.io import WaveformEvent


def analyze_events(
    events: Iterable[WaveformEvent],
) -> list[dict]:
    """
    Analyze a sequence of waveform events.

    Parameters
    ----------
    events : iterable of WaveformEvent
        ROOT waveform events.

    Returns
    -------
    list of dict
        Event-analysis results.
    """

    results = []

    for event in events:
        event_results = analyze_event(
            file_name=event.file_path.name,
            event_index=event.event_index,
            time=event.time,
            channel1=event.channel1,
            channel2=event.channel2,
        )

        results.extend(event_results)

    return results
