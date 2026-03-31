import logging
import os
import posixpath
from stat import S_ISDIR

import dotenv
import mlflow
import paramiko

logger = logging.getLogger(__name__)


class MLFLowTools:
    def __init__(self, server="10.151.0.57", username="riskusers"):
        # Load .env file
        dotenv.load_dotenv()
        self.server = server
        self.username = username
        self.ssh_passphrase = os.getenv("SSH_PASSPHRASE")
        self.ssh_key_path = os.getenv("SSH_RISK_KEY_PATH")
        self.ssh, self.sftp = self.create_connection()

    def create_connection(self):
        try:
            # Load SSH key
            ssh_key = paramiko.RSAKey(filename=os.path.expanduser(self.ssh_key_path), password=self.ssh_passphrase)

            # Connect to the server via SSH using the private key
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(self.server, username=self.username, pkey=ssh_key)

            # Open SFTP session
            sftp = ssh.open_sftp()
            return ssh, sftp
        except Exception as e:
            logger.error(f"Error creating connection: {e}")
            raise e

    def log_artifact(self, local_path: str, experiment_id=None, run_id=None, remote_server_dir="mlflow_artifacts"):
        logger.info(f"Logging artifact: {local_path}")
        if self.server == "localhost" or self.server == "127.0.0.1":
            mlflow.log_artifact(local_path)
        else:
            self.send_artifact_to_remote(
                experiment_id=experiment_id, run_id=run_id, local_filepath=local_path,
                remote_server_dir=remote_server_dir,
            )

    def send_artifact_to_remote(self, experiment_id: str, run_id: str, local_filepath: str,
                                remote_server_dir="mlflow_artifacts") -> str:
        """
        Sends an artifact file to the mlflow_artifacts folder on the remote server using an SSH key.

        :param experiment_id: MLflow experiment ID
        :param run_id: MLflow run ID
        :param local_filepath: Path to the artifact file on the local machine
        :param remote_server_dir: Directory on the remote server
        """
        base = f"/home/{self.username}/{remote_server_dir}/{experiment_id}/{run_id}"
        artifacts_dir = posixpath.join(base, "artifacts")
        remote_filepath = posixpath.join(artifacts_dir, os.path.basename(local_filepath))

        try:
            # Ensure the experiment directory exists
            logger.debug(f"Make directory if not exists: {artifacts_dir}")
            self.ensure_remote_dir(artifacts_dir)

            # Upload the artifact
            self.sftp.put(local_filepath, remote_filepath)
            logger.info(f"Artifact {local_filepath} uploaded to {remote_filepath}")

            return remote_filepath

        except Exception as e:
            logger.error(f"Error uploading artifact: {e}")
            raise

    def ensure_remote_dir(self, remote_dir: str):
        """
        Recursively ensure 'remote_dir' exists (POSIX); no-ops if it already exists.
        Works with absolute paths like '/home/user/mlflow_artifacts/.../artifacts'.
        """
        # Normalize and split into parts
        remote_dir = posixpath.normpath(remote_dir)
        if remote_dir in ('', '/'):
            return  # nothing to do

        parts = remote_dir.strip('/').split('/')
        cur = '/'

        for part in parts:
            cur = posixpath.join(cur, part) if cur != '/' else '/' + part
            try:
                st = self.sftp.stat(cur)
                if not S_ISDIR(st.st_mode):
                    raise NotADirectoryError(f"Remote path exists but is not a directory: {cur}")
            except FileNotFoundError:
                # Directory doesn't exist -> create it
                try:
                    self.sftp.mkdir(cur)  # optionally: sftp.mkdir(cur, mode=0o755)
                except IOError:
                    # Handle a race where another process created it between stat and mkdir
                    st2 = self.sftp.stat(cur)
                    if not S_ISDIR(st2.st_mode):
                        raise

    def close_connection(self):
        if self.sftp:
            self.sftp.close()
        if self.ssh:
            self.ssh.close()
