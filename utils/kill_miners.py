import re
import subprocess as proc


def kill_miners():
    res = proc.run(["pidof", "ethminer"], stdout=proc.PIPE)
    pid = res.stdout.decode("utf-8")
    is_mining = res.returncode == 0

    if is_mining:
        regexp = re.compile("\d+")
        pid = regexp.search(pid)[0]
        proc.run(["kill", "-SIGKILL", pid])
        print("Killing miner")