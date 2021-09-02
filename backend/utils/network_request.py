import requests
from requests_toolbelt import sessions
from requests.packages.urllib3.util.retry import Retry

from requests.adapters import HTTPAdapter

DEFAULT_TIMEOUT = 5  # seconds


class TimeoutHTTPAdapter(HTTPAdapter):
    def __init__(self, *args, **kwargs):
        self.timeout = DEFAULT_TIMEOUT
        if "timeout" in kwargs:
            self.timeout = kwargs["timeout"]
            del kwargs["timeout"]
        super().__init__(*args, **kwargs)

    def send(self, request, **kwargs):
        timeout = kwargs.get("timeout")
        if timeout is None:
            kwargs["timeout"] = self.timeout
        return super().send(request, **kwargs)


def request_client_generator(base_url: str = "") -> requests.Session:
    """
    Generic request instance with base_url, retry and exponential backoff logic included
    Should be imported only once per module with a different base_url for each microservice this connects to
    """
    http = requests.Session()
    http.hooks["response"] = [lambda response, *args, **kwargs: response.raise_for_status()]
    http = sessions.BaseUrlSession(base_url=base_url)

    adapter = TimeoutHTTPAdapter(timeout=2.5)
    http.mount("https://", adapter)
    http.mount("http://", adapter)

    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    http.mount("https://", TimeoutHTTPAdapter(max_retries=retries))

    return http
