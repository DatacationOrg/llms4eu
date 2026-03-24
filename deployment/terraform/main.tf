provider "azurerm" {
  features {}
  subscription_id = var.subscription_id
}

provider "github" {
  owner = "DatacationOrg"
}

provider "azuread" {
  tenant_id = var.tenant_id
}

resource "azurerm_resource_group" "main" {
  name     = "${var.deployment_name}-rg"
  location = var.resource_group_location
}

variable "resource_group_location" {
  description = "The Azure region for the resource group"
  type        = string
}

variable "tenant_id" {
  description = "The Azure AD tenant ID"
  type        = string
}

variable "subscription_id" {
  description = "The Azure subscription ID"
  type        = string
}

variable "deployment_name" {
  description = "The name of the deployment, used for naming resources"
  type        = string
}

terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.21.1"
    }
    azuread = {
      source  = "hashicorp/azuread"
      version = "~> 3.1.0"
    }
    github = {
      source  = "integrations/github"
      version = "~> 6.6.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.7.1"
    }
  }
}

resource "local_file" "outputs" {
  content = jsonencode({
    DEPLOYMENT_NAME                       = "dev"
    APPLICATIONINSIGHTS_CONNECTION_STRING = "${azurerm_application_insights.main.connection_string}"
    APP_SERVICE_NAME                      = "${azurerm_linux_web_app.webapp.name}"
  })
  filename = "${path.module}/terraform-outputs.json"
}
