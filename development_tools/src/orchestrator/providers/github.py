from __future__ import annotations

from typing import Optional, Dict, Any
import httpx

from ..config import settings


class GitHubClient:
    def __init__(self, token: Optional[str] = None, dry_run: Optional[bool] = None):
        self.token = token or settings.github_token
        self.dry_run = settings.github_dry_run if dry_run is None else dry_run
        self._http = httpx.Client(base_url="https://api.github.com", headers=self._headers(), timeout=60)

    def _headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/vnd.github+json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def create_pull_request(self, owner: str, repo: str, title: str, head: str, base: str, body: str) -> dict:
        if self.dry_run:
            return {"number": 0, "html_url": f"https://github.com/{owner}/{repo}/pull/0"}
        resp = self._http.post(f"/repos/{owner}/{repo}/pulls", json={
            "title": title,
            "head": head,
            "base": base,
            "body": body,
        })
        resp.raise_for_status()
        data = resp.json()
        return {"number": int(data["number"]), "html_url": data.get("html_url")}

    def comment_on_issue(self, owner: str, repo: str, issue_number: int, body: str) -> None:
        if self.dry_run:
            return
        resp = self._http.post(f"/repos/{owner}/{repo}/issues/{issue_number}/comments", json={"body": body})
        resp.raise_for_status()

    def add_labels(self, owner: str, repo: str, issue_number: int, labels: list[str]) -> None:
        if self.dry_run:
            return
        resp = self._http.post(f"/repos/{owner}/{repo}/issues/{issue_number}/labels", json={"labels": labels})
        resp.raise_for_status()

    def merge_pull_request(self, owner: str, repo: str, pr_number: int, merge_method: str = "merge") -> dict:
        if self.dry_run:
            return {"merged": True, "sha": "dryrun"}
        resp = self._http.put(f"/repos/{owner}/{repo}/pulls/{pr_number}/merge", json={"merge_method": merge_method})
        if resp.status_code == 405:
            return {"merged": False, "message": "Not mergeable"}
        resp.raise_for_status()
        return resp.json()


