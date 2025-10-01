const runId = 1; // dynamic per workflow run
const evtSource = new EventSource(`/stream/${runId}`);

evtSource.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log("Step update:", data);
    // TODO: Update UI dynamically
};
