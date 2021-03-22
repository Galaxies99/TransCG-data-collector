# NetRelay
NetRelay is a simple network relay tool forwarding the shell command to the destination terminal.

## Usage

**For relay terminal**

```bash
python3 relay.py --src=<sourceAddr>
```

**For client terminal**

```bash
python3 client.py --dst=<destinationAddr>[ --error]
```

- `--error` will display `stderr` message on the screen.

**For API use**

```python
import netrelay.client as nr_client
```

- Use `nr_client.start(<ipPortAddr>)` and `nr_client.close()` to start the connection to the relay server and close the connection to the relay server. `start` will return two values `s, id` representing the remote relay server `<server>` and the index of the current client on the remote relay server.

- Use `nr_client.exec_cmd(<server>, <command>)` to execute the `<command>` remotely on relay server `<server>`. `exec_cmd` will return two values `res, err` representing the result of the execution and the error message of the execution respectively.
- Use `nr_client.exec_cmd_and_save(<server>, <command>, <resultDir>[, <errorMessageDir>])` to execute the `<command>` remotely on relay server `<server>`, and save the result in `<resultDir>`, save the error message in `<errorMessageDir>`.

Here is an example.

```python
import netrelay.client as nr_client
s, id = nr_client.start(('127.0.0.1', 2333))
res, err = nr_client.exec_cmd(s, 'curl -L www.linux.com')
nr_client.exec_cmd_and_save(s, 'curl -L www.linux.com', 'linux.html')
nr_client.close()
```

## Relay Tools for PST SDK REST API

We modified the project to adapt to the PST REST API, and this tools enable users to call the PST REST API on a Windows PST-installed computer from a remote computer (for example, a Linux client which does not have PST support). See [PST SDK Documentation](http://files.ps-tech.com/pst/docs/5.0.1/SDK/rest.html) for details about the PST REST API. Since the PST REST API includes streaming output, we use `pycurl` to re-implement the relay server. The client remains the same.

**For relay terminal**

```bash
python3 relay_pstrest.py --src=<sourceAddr>
```

**For client terminal**

```bash
python3 client_pstrest.py --dst=<destinationAddr>[ --error]
```

You will be asked to input the filename that stores the result. Since this tool is designed for PST SDK REST API, the result will be saved in `result` directory with a `.json` format.

**Note**. This tool only supports PST REST API currently, but may not support other `curl`-based APIs.

## To-dos

- [x] mini-shell client
- [x] client API support
- [x] Support PST SDK REST API
- [ ] Support full `curl` method

