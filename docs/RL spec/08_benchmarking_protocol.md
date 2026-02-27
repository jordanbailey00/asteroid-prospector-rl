## 7) Quick benchmarking protocol (so you can compare PPO runs)

Run each bot for `N=1000` episodes across random seeds, log:
- mean/median net profit
- survival rate
- profit per tick
- overheat ticks / episode
- pirate encounters / episode

Then compare:
- PPO (random init) vs greedy miner (should beat it)
- PPO vs cautious scanner (should learn less deaths)
- PPO vs market timer (should learn timing / slippage behavior)
