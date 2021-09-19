import os
from dataclasses import dataclass, field
from typing import Dict, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow, InstalledAppFlow
from googleapiclient.discovery import Resource, build

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
    _credentials: Optional[Credentials] = field(init=False)
    _drive_service: Resource = field(init=False)

    def __post_init__(self) -> None:
        self.flow = Flow.from_client_secrets_file(
            path_to_client_secret, SCOPES.get(self.mode)
        )
        self.flow.redirect_uri = "http://localhost:8080/"
        self._credentials = self._get_credentials()
        self._drive_service = build(
            "drive", "v3", credentials=self._credentials
        )

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

    def list_files(self) -> None:
        """Lists the files in google drive"""
        drive_service = getattr(self, "_drive_service")
        results = (
            drive_service.files()
            .list(pageSize=10, fields="nextPageToken, files(id, name)")
            .execute()
        )
        files_in_drive = results.get("files", [])

        if not files_in_drive:
            print("Noting found")
        else:
            print("Files:")
            for file in files_in_drive:
                print(u"{0} ({1})".format(file["name"], file["id"]))

    def download_all_files_from_drive(self) -> None:
        pass

    def download_file_from_drive(
        self, file_name: str, target_path: Optional[str] = None
    ) -> None:
        pass

    def export_file_to_drive(self, file_name: str, source_path: str) -> None:
        pass
