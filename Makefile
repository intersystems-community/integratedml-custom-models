# IntegratedML Custom Models - Development Automation
# Simplified workflow for IRIS + local development

.PHONY: help setup start stop clean install test format lint notebooks

# Default target
help:
	@echo "IntegratedML Custom Models - Available Commands:"
	@echo ""
	@echo "Setup & Environment:"
	@echo "  setup     - Complete project setup (dependencies + IRIS)"
	@echo "  install   - Install Python dependencies only"
	@echo "  start     - Start IRIS database"
	@echo "  stop      - Stop IRIS database"
	@echo "  clean     - Clean up containers and volumes"
	@echo ""
	@echo "Development:"
	@echo "  notebooks - Open notebooks directory in VS Code"
	@echo "  test      - Run all tests"
	@echo "  format    - Format code with black"
	@echo "  lint      - Run linting checks"
	@echo ""
	@echo "Quick Start:"
	@echo "  make setup && make notebooks"

# Environment setup
setup: install start
	@echo "✅ Setup complete! Run 'make notebooks' to start exploring."

install:
	@echo "📦 Installing Python dependencies with uv..."
	@if command -v uv >/dev/null 2>&1; then \
		uv sync; \
	else \
		echo "⚠️  uv not found, falling back to pip..."; \
		pip install -r requirements.txt; \
	fi
	@echo "✅ Dependencies installed"

install-uv:
	@echo "🚀 Installing uv (ultra-fast Python package manager)..."
	curl -LsSf https://astral.sh/uv/install.sh | sh
	@echo "✅ uv installed! Restart your shell or run: source ~/.cargo/env"

# IRIS database management
start:
	@echo "🚀 Starting IRIS database..."
	docker-compose up -d iris
	@echo "⏳ Waiting for IRIS to be ready..."
	@timeout 120 bash -c 'until docker-compose exec iris iris session iris -U USER "w \"Database ready\""; do sleep 2; done'
	@echo "✅ IRIS database is ready!"
	@echo "   Management Portal: http://localhost:52773/csp/sys/UtilHome.csp"
	@echo "   Database Port: localhost:1972"

stop:
	@echo "🛑 Stopping IRIS database..."
	docker-compose stop iris
	@echo "✅ IRIS stopped"

restart: stop start

# Development tools
notebooks:
	@echo "📓 Opening notebooks in VS Code..."
	code notebooks/ demos/

test:
	@echo "🧪 Running tests..."
	@pytest demos/*/tests/ -v --tb=short
	@echo "✅ Tests completed"

format:
	@echo "🎨 Formatting code..."
	black .
	@echo "✅ Code formatted"

lint:
	@echo "🔍 Running linting..."
	flake8 . --max-line-length=88 --extend-ignore=E203,W503
	mypy shared/ --ignore-missing-imports
	@echo "✅ Linting completed"

# Cleanup
clean:
	@echo "🧹 Cleaning up..."
	docker-compose down -v
	docker system prune -f
	@echo "✅ Cleanup completed"

# Demo-specific targets
demo-credit:
	@echo "💳 Running Credit Risk demo..."
	python run_credit_risk_demo.py

demo-fraud:
	@echo "🔒 Running Fraud Detection demo..."
	python run_fraud_detection_demo.py

demo-sales:
	@echo "📈 Running Sales Forecasting demo..."
	python run_sales_forecasting_demo.py

demo-dna:
	@echo "🧬 Running DNA Similarity demo..."
	python run_dna_similarity_demo.py

demo-ai-functions:
	@echo "🤖 Running AI Functions demo (sentiment, classify, summarize, translate, embed, extract, complete)..."
	python run_ai_functions_demo.py

# Quick demo runner
demos: start
	@echo "🎬 Running all demos..."
	@$(MAKE) demo-credit
	@$(MAKE) demo-fraud
	@$(MAKE) demo-sales
	@$(MAKE) demo-dna
	@$(MAKE) demo-ai-functions
	@echo "✅ All demos completed!"

# Status check
status:
	@echo "📊 System Status:"
	@echo ""
	@docker-compose ps
	@echo ""
	@if docker-compose exec iris iris session iris -U USER "w \"IRIS Status: Online\"" 2>/dev/null; then \
		echo "✅ IRIS: Online"; \
	else \
		echo "❌ IRIS: Offline"; \
	fi

# Development utilities
dev-setup: install
	@echo "🛠️  Setting up development environment..."
	pip install pre-commit
	pre-commit install
	@echo "✅ Development tools configured"

# Container logs
logs:
	docker-compose logs -f iris

# Database initialization
init-db: start
	@echo "🗃️  Initializing database with sample data..."
	python setup_iris_integratedml.py
	@echo "✅ Database initialized"