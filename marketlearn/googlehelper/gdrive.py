import os
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow, InstalledAppFlow

# set scope of drive here
SCOPES: Dict[str, str] = {
    "readonly": "https://www.googleapis.com/auth/drive.metadata.readonly"
}

path = os.path.dirname(os.getcwd())
path_to_client_secret = os.path.join(path, "client_secret.json")


@dataclass
class GDrive:
    mode: str = "readonly"
    flow: Flow = field(init=False)
    credentials: Optional[Credentials] = field(init=False)

    def __post_init__(self) -> None:
        self.flow = Flow.from_client_secrets_file(
            path_to_client_secret, SCOPES.get(self.mode)
        )
        self.flow.redirect_uri = "http://localhost:8080/"
        self.credentials = self._get_credentials()

    def _get_credentials(self) -> Credentials:
        creds: Optional[Credentials] = None
        if os.path.exists("token.json"):
            creds = Credentials.from_authorized_user_file(
                "token.json", SCOPES.get(self.mode)
            )

        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    path_to_client_secret, SCOPES.get(self.mode)
                )
                creds = flow.run_local_server(port=8080)

        # Save the credentials for the next run
        with open("token.json", "w") as token:
            token.write(creds.to_json())

        return creds
