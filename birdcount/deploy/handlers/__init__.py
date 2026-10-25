"""What the consumer does with each batch of results: the alarm decision
(`engine`) and the actions and records driven by it (SMS, speakers, smart
plugs, the count log). All share `base.Handler` and run on the consumer
thread; anything slow (disk, network, devices) is handed to their own threads
so a batch never waits on I/O."""
