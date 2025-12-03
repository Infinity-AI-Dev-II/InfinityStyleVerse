--Overall module setup information
Design pattern used: 
CQRS Pattern

Purpose:
TaskPulseOS is the live “heartbeat monitor” and control plane for InfinityBrain™. It tracks 
every workflow, task, and step running across NeuroCoreOS and FlowGateOS, exposes real
time status and metrics, enables operator actions (pause/resume/cancel/retry), and streams 
events to UIs and downstream learners (EchoOS). Think: Durable telemetry + run state + 
control surface for all AI pipelines. 
