# NetRelay for PST SDK REST API

NetRelay is a simple network relay tool forwarding the shell command to the destination terminal. We modified the project to adapt to the PST REST API, and this tools enable users to call the PST REST API on a Windows PST-installed computer from a remote computer (for example, a Linux client which does not have PST support). See [PST SDK Documentation](http://files.ps-tech.com/pst/docs/5.0.1/SDK/rest.html) for details about the PST REST API. Since the PST REST API includes streaming output, we use `pycurl` to re-implement the relay server. The client remains the same.

## Usage

### For relay terminal

```bash
python3 relay_cmd.py --src=<sourceAddr>
```

or

```bash
python3 stream.py --src=<sourceAddr>
python3 relay_stream.py --src=<sourceAddr>
```

The former one allows `exit` and formal `curl` command input, netrelay will execute the command on the remote computer, and send back the results.

The latter one only allows `exit` and `GetTracker` command input. When receiving `GetTracker`, netrelay will get the data from the tracker stream created by `stream.py`, and send back the data.

If you encounter with an `ImportError` exception, try to add `-m` to run the program as a python module.

### For client terminal

```bash
python3 client.py --dst=<destinationAddr>[ --error]
```

You will be asked to input the filename that stores the result. Since this tool is designed for PST SDK REST API, the result will be saved in `results` directory with a `.json` format.

**Note**. This tool only supports PST REST API currently, but may not support other `curl`-based APIs.

## API Usage

```python
import netrelay.client as nr_client
```

- `nr_client.start(dst_addr)`: Start the client process, return a socket object `s` and its integer `id`;
- `nr_client.close(s)`: Close the client process of socket object `s`;
- `nr_client.exec_cmd(s, cmd)`: Execute `cmd` in socket object `s`;
- `nr_client.exec_cmd_and_save(s, cmd, res_dir, display=False)`: Execute `cmd` in socket object `s`, and save the results in `res_dir`, display the results on screen if `display = True`.
