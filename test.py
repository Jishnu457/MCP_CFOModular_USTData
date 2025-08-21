from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

# Your Key Vault URL
key_vault_url = "https://cybfnai.vault.azure.net/"

# Create credential and client
credential = DefaultAzureCredential()
client = SecretClient(vault_url=key_vault_url, credential=credential)

# Retrieve a secret
secret = client.get_secret("azure-client-id")
print(secret.value)