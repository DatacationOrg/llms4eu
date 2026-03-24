data "azurerm_client_config" "current" {}

data "azuread_user" "current_user" {
  object_id = data.azurerm_client_config.current.object_id
}

resource "random_uuid" "oauth2_permission_scope" {
}

resource "azuread_application" "app_deploy" {
  display_name = "${var.deployment_name}-app-deploy"
  owners       = [data.azuread_user.current_user.object_id]
  lifecycle {
    ignore_changes = [
      owners
    ]
  }
}

resource "azuread_service_principal" "app_deploy" {
  client_id    = azuread_application.app_deploy.client_id
  owners       = [data.azuread_user.current_user.object_id]
  use_existing = true
  lifecycle {
    ignore_changes = [
      owners
    ]
  }
}

resource "azuread_service_principal_password" "app_deploy" {
  service_principal_id = azuread_service_principal.app_deploy.id
  depends_on           = [azuread_service_principal.app_deploy]
}

resource "azurerm_role_assignment" "acr_push" {
  principal_id         = azuread_service_principal.app_deploy.object_id
  role_definition_name = "AcrPush"
  scope                = azurerm_container_registry.main.id
  depends_on = [
    azuread_service_principal.app_deploy,
    azurerm_container_registry.main
  ]
}

resource "azurerm_role_assignment" "app_restart" {
  principal_id         = azuread_service_principal.app_deploy.object_id
  role_definition_name = "Contributor"
  scope                = azurerm_linux_web_app.webapp.id
  depends_on = [
    azuread_service_principal.app_deploy,
    azurerm_linux_web_app.webapp
  ]
}

resource "azurerm_role_assignment" "acr_pull_image" {
  principal_id         = azurerm_linux_web_app.webapp.identity[0].principal_id
  role_definition_name = "AcrPull"
  scope                = azurerm_container_registry.main.id
  depends_on = [
    azurerm_linux_web_app.webapp
  ]
}
