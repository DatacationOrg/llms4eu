

resource "azurerm_container_registry" "main" {
  name                = "${var.deployment_name}acr"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = "Standard"
  admin_enabled       = true
}

resource "azurerm_container_registry_task" "purge_task" {
  container_registry_id = azurerm_container_registry.main.id
  name                  = "${var.deployment_name}-acr-purge-task"
  agent_setting {
    cpu = 2
  }
  base_image_trigger {
    name                        = "defaultBaseimageTriggerName"
    type                        = "Runtime"
    update_trigger_payload_type = "Default"
  }
  encoded_step {
    task_content = <<EOF
version: v1.1.0
steps:
  - cmd: acr purge --filter '${var.deployment_name}:^sha-.*' --untagged --ago 2d --keep 3
    disableWorkingDirectoryOverride: true
    timeout: 3600
EOF
  }
  platform {
    architecture = "amd64"
    os           = "Linux"
  }
  timer_trigger {
    name     = "daily"
    schedule = "0 0 * * *"
  }
  lifecycle {
    ignore_changes = [
      encoded_step[0].task_content
    ]
  }
}

data "github_repository" "main" {
  name = "lllms4eu"
}

resource "github_actions_secret" "acr_url" {
  repository      = data.github_repository.main.name
  secret_name     = "ACR_URL" # pragma: allowlist secret
  plaintext_value = azurerm_container_registry.main.login_server
}

resource "github_actions_secret" "azure_client_id" {
  repository      = data.github_repository.main.name
  secret_name     = "AZURE_CLIENT_ID" # pragma: allowlist secret
  plaintext_value = azuread_application.app_deploy.client_id
}

resource "github_actions_secret" "azure_client_secret" {
  repository      = data.github_repository.main.name
  secret_name     = "AZURE_CLIENT_SECRET" # pragma: allowlist secret
  plaintext_value = azuread_service_principal_password.app_deploy.value
}

resource "github_actions_secret" "azure_credentials" {
  repository  = data.github_repository.main.name
  secret_name = "AZURE_CREDENTIALS" # pragma: allowlist secret
  plaintext_value = jsonencode({
    clientId       = azuread_application.app_deploy.client_id
    clientSecret   = azuread_service_principal_password.app_deploy.value
    tenantId       = data.azurerm_client_config.current.tenant_id
    subscriptionId = var.subscription_id
  })
}

resource "azurerm_service_plan" "main" {
  name                = "app-service-plan-${azurerm_resource_group.main.name}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  os_type             = "Linux"
  sku_name            = "B2"
}

resource "azurerm_application_insights" "main" {
  name                = "${var.deployment_name}-appi"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  application_type    = "web"
}


resource "azurerm_linux_web_app" "webapp" {
  name                = "${var.deployment_name}-webapp"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  service_plan_id     = azurerm_service_plan.main.id
  https_only          = true

  site_config {
    always_on                         = true
    container_registry_use_managed_identity = true
    # TODO include health checks
  }

  app_settings = {
    WEBSITES_PORT                             = "8000"
    WEBSITES_ENABLE_APP_SERVICE_STORAGE       = "false"
    DEPLOYMENT_NAME                           = "prod"
    DOCKER_ENABLE_CI                          = "true"
    APPLICATIONINSIGHTS_CONNECTION_STRING     = azurerm_application_insights.main.connection_string
    APP_SERVICE_NAME                          = "${var.deployment_name}-webapp"
  }


  depends_on = [
    azurerm_service_plan.main,
  ]

  identity {
    type = "SystemAssigned"
  }
}
