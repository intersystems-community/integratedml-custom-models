# Existing IntegratedML Documentation Analysis

**Purpose**: Understand the existing IntegratedML/AutoML documentation foundation that the doc team will build upon for Custom Models documentation.

**Source**: docs.intersystems.com
**Date Analyzed**: 2025-01-12

## Executive Summary

The doc team has a **strong, well-established foundation** in the existing IntegratedML documentation. Custom Models will be positioned as an **extension** of IntegratedML, not a replacement. The existing docs provide excellent patterns to follow for SQL syntax, concepts, and user guidance.

## Existing Documentation Structure

### Main Documentation Guides

#### 1. "Using IntegratedML" Guide
**URL Pattern**: `docs.intersystems.com/.../KEY=GIML_*`

**Key Pages**:
- **Introduction to IntegratedML** (`GIML_Intro`)
  - Purpose and value proposition
  - Machine learning concepts (regression vs classification, training, features/labels)
  - How IntegratedML differs from traditional ML

- **IntegratedML Basics** (`GIML_Basics`)
  - Core SQL commands
  - Basic workflows
  - Examples

- **ML Configurations** (`GIML_Configuration`)
  - Configuring ML providers
  - Setting default configurations
  - Provider-specific settings

- **Providers** (`GIML_Configuration_Providers`)
  - %AutoML (system default)
  - DataRobot
  - H2O
  - PMML (pre-trained model import)

- **SQL Syntax**
  - CREATE MODEL
  - TRAIN MODEL
  - VALIDATE MODEL
  - PREDICT()
  - PROBABILITY()
  - CREATE ML CONFIGURATION
  - SET ML CONFIGURATION

#### 2. "AutoML Reference" Guide
**URL Pattern**: `docs.intersystems.com/.../KEY=GAUTOML_*`

**Key Pages**:
- **Introduction to AutoML** (`GAUTOML_Intro`)
  - Technical deep-dive into AutoML engine
  - How AutoML works internally
  - Architecture and algorithms

**Content Includes**:
- Feature engineering automation
- Model selection process
- Training parameters
- Performance optimization

### Documentation Versions

Documentation exists for multiple IRIS versions:
- IRIS 2025.2 (latest)
- IRIS 2025.1
- IRIS for Health 2024.3
- IRIS for Health 2024.1
- IRIS 2022.3
- IRIS 2022.1

## Key Concepts from Existing Docs

### 1. IntegratedML Positioning

**Official Description**:
> "IntegratedML is a feature within InterSystems IRIS® data platform which allows you to use automated machine learning functions directly from SQL to create and use predictive models."

**Key Value Propositions**:
- ✅ All-SQL interface - "Build and train ML models using intuitive custom SQL commands"
- ✅ Turnkey solution - "Nothing to install, no packages or programming languages to learn"
- ✅ Modular - "Leverages best-of-breed open source and proprietary AutoML frameworks"
- ✅ No ML expertise required - "Without expertise in feature engineering or ML algorithms"
- ✅ Quick time to value - "Considerably reduces barrier to entry"
- ✅ Complements data scientists - "Not meant to replace data scientists, but complement them"

### 2. Provider Architecture

**Current Provider Model**:
```
IntegratedML (SQL Interface)
    ↓
ML Configuration (Named configuration)
    ↓
Provider (Execution engine)
    - %AutoML (default)
    - DataRobot
    - H2O
    - PMML
```

**Default Configuration**:
- %AutoML is system-default upon installation
- Users can create custom ML configurations pointing to different providers

### 3. SQL Command Patterns

**Existing SQL Syntax**:

```sql
-- Create model definition
CREATE MODEL ModelName
PREDICTING (target_column)
FROM TableName

-- Train with provider-specific parameters
TRAIN MODEL ModelName
USING {"seed": 3, "IsRegression": 1}

-- Validate model
VALIDATE MODEL ModelName
FROM TestTable

-- Make predictions
SELECT id, feature1,
       PREDICT(ModelName) as prediction,
       PROBABILITY(ModelName FOR 1) as confidence
FROM NewData

-- Configure ML provider
CREATE ML CONFIGURATION config_name
PROVIDER DataRobot

SET ML CONFIGURATION config_name
```

### 4. AutoML Features (Existing Provider)

**What %AutoML Provides**:

1. **Automatic Feature Engineering**:
   - Column type classification
   - Feature elimination (remove redundancy)
   - One-hot encoding of categorical features
   - Missing value imputation
   - Time-based feature extraction (hours/days/months/years)

2. **Automatic Model Selection**:
   - Determines regression vs classification
   - Samples large datasets for selection speed
   - Uses Monte Carlo cross-validation
   - Selects best model based on scoring metrics
   - Trains winning model on full dataset

3. **Training Parameters**:
   - `seed` - Random seed
   - `IsRegression` - Force regression (1) or classification (0)
   - Provider-specific parameters

4. **Installation**:
   - Python package: `intersystems-iris-automl` or `intersystems-iris-automl-tf`
   - Requires Python 3.11+
   - Installed via pip

### 5. Third-Party Provider Integration

**DataRobot**:
- Business relationship required
- Uses DataRobot API
- HTTP requests for training
- `quickrun` parameter defaults to true

**H2O**:
- Mentioned as third-party option
- Integration details in docs

**PMML**:
- Import pre-trained models
- No training needed (model already trained)
- Uses USING clause to specify PMML file
- Can select specific model from multi-model PMML files

## How Custom Models Fits In

### Extension, Not Replacement

**Custom Models adds a NEW provider type** to the existing provider architecture:

```
IntegratedML (SQL Interface)
    ↓
ML Configuration (Named configuration)
    ↓
Provider (Execution engine)
    - %AutoML (automated, no code)          ← EXISTING
    - DataRobot (third-party AutoML)        ← EXISTING
    - H2O (third-party AutoML)              ← EXISTING
    - PMML (pre-trained model import)       ← EXISTING
    - Custom Models (Python code control)   ← NEW in 2025.2
```

### Key Differentiators

| Aspect | AutoML (Existing) | Custom Models (New) |
|--------|-------------------|---------------------|
| **User Control** | Automated | Full control |
| **Code Required** | None (SQL only) | Python model class |
| **Feature Engineering** | Automatic | Custom Python code |
| **Model Selection** | Automatic | Developer chooses |
| **Expertise Level** | None required | Python + ML knowledge |
| **Use Case** | Quick models, no expertise | Custom logic, domain-specific |
| **Libraries** | Built-in AutoML | Any scikit-learn compatible |
| **Third-Party** | DataRobot, H2O | Prophet, LightGBM, etc. |

### SQL Syntax Evolution

**AutoML (Existing)**:
```sql
TRAIN MODEL my_model
-- No USING clause = use default %AutoML
-- OR simple parameters:
USING {"seed": 3}
```

**Custom Models (New in 2025.2)**:
```sql
TRAIN MODEL my_model
USING {
    "model_name": "CustomCreditRiskClassifier",
    "path_to_classifiers": "/opt/iris/mgr/python/custom_models/classifiers",
    "user_params": {
        "enable_debt_ratio": 1,
        "enable_risk_scoring": 1
    }
}
```

**Key Addition**: JSON USING clause with:
- `model_name` - Python class name
- `path_to_classifiers` - File path to custom model
- `user_params` - Custom model parameters

## Documentation Patterns to Follow

### 1. Conceptual Introduction Pattern

**From "Introduction to IntegratedML"**:
- ✅ Start with "Purpose" - Why does this exist?
- ✅ Define target audience clearly
- ✅ Explain basic ML concepts (don't assume knowledge)
- ✅ Position relative to traditional approaches
- ✅ Set expectations (not meant to replace X, but complement)

**Apply to Custom Models**:
- Purpose: Full control for custom ML logic
- Audience: Python developers, data scientists
- Position: Extends AutoML with custom code capability
- Expectation: Requires Python + ML knowledge

### 2. Basics/Getting Started Pattern

**From "IntegratedML Basics"**:
- ✅ Show complete workflow start-to-finish
- ✅ Simple example with real SQL
- ✅ Explain each SQL command
- ✅ Show expected output
- ✅ Link to detailed references

**Apply to Custom Models**:
- Show: Python class → SQL CREATE → TRAIN → PREDICT
- Simple example: Credit risk or similar
- Explain new USING syntax
- Show prediction output
- Link to API reference for custom classes

### 3. Configuration Pattern

**From "ML Configurations"**:
- ✅ Explain configuration concept
- ✅ Show how to create configurations
- ✅ Show how to set default
- ✅ Explain when to use multiple configurations

**Apply to Custom Models**:
- Configuration for custom model path
- Creating configuration for custom models
- Switching between AutoML and Custom Models
- Use cases for each

### 4. Provider-Specific Pattern

**From "Providers" page**:
- ✅ One section per provider
- ✅ Installation requirements
- ✅ Training parameters
- ✅ Special features
- ✅ Example SQL with provider

**Apply to Custom Models**:
- "Custom Models" section in Providers
- Installation: Python model classes
- Training parameters: USING clause reference
- Special features: Full Python control
- Example with custom class

### 5. Reference Documentation Pattern

**From "AutoML Reference"**:
- ✅ Separate technical reference guide
- ✅ Deep-dive into how it works
- ✅ Architecture diagrams
- ✅ Algorithm details
- ✅ Performance characteristics

**Apply to Custom Models**:
- Custom Models Reference guide
- Base class architecture
- Model lifecycle (fit/predict/validate)
- Integration with IRIS
- Performance considerations

## Documentation Gaps to Fill

### 1. User Guide Updates

**Existing "Using IntegratedML" Needs**:
- [ ] New section: "Introduction to Custom Models"
- [ ] Updated "Providers" page: Add Custom Models section
- [ ] Updated SQL syntax examples: Show USING clause with JSON
- [ ] New page: "Creating Custom Model Classes"
- [ ] Updated "When to Use" guidance: AutoML vs Custom Models decision tree

### 2. New Reference Guide

**"Custom Models Reference" (New Guide)**:
- [ ] Architecture: Base classes and inheritance
- [ ] API Reference: IntegratedMLBaseModel, ClassificationModel, RegressionModel
- [ ] Required Methods: fit(), predict(), _validate_parameters()
- [ ] Model Lifecycle: Creation → Training → Validation → Prediction
- [ ] Integration: How models execute within IRIS
- [ ] Examples: Step-by-step custom model creation

### 3. Migration/Upgrade Documentation

**"Upgrading from AutoML to Custom Models"**:
- [ ] When to stay with AutoML
- [ ] When to migrate to Custom Models
- [ ] Side-by-side comparison examples
- [ ] Migration checklist

### 4. SQL Reference Updates

**SQL Reference Manual Updates**:
- [ ] UPDATE: CREATE MODEL - Document JSON USING clause
- [ ] UPDATE: TRAIN MODEL - Add custom model parameters
- [ ] NEW: Custom model parameter reference
- [ ] Examples: Custom Models in SQL reference

## Documentation Strategy for Custom Models

### Phase 1: Extend Existing Guides (Minimal Disruption)

**Goal**: Integrate Custom Models into existing documentation structure

1. **Update "Using IntegratedML" Guide**:
   - Add "Custom Models" to Providers page (alongside AutoML, DataRobot, H2O, PMML)
   - Add brief introduction positioning Custom Models
   - Update SQL syntax examples to show JSON USING clause
   - Add "When to Use Custom Models" section

2. **Add Brief Quick Start**:
   - 5-minute example showing custom model creation
   - Link to full reference in new guide

**Timeline**: For 2026.1 GA release (minimal scope)

### Phase 2: Comprehensive Reference Guide (Full Documentation)

**Goal**: Provide complete technical documentation for Custom Models

3. **Create "Custom Models Reference" Guide**:
   - Full API reference for base classes
   - Architecture documentation
   - Model lifecycle documentation
   - Integration details
   - Performance tuning
   - Best practices

4. **Create Tutorial/Examples Section**:
   - Step-by-step tutorials for common scenarios
   - Demo application walkthroughs
   - Advanced examples

**Timeline**: For 2026.2 or later (full scope)

### Phase 3: Integration & Cross-Linking (Polish)

**Goal**: Seamless navigation between guides

5. **Cross-Reference Updates**:
   - Link from AutoML docs to Custom Models (for advanced users)
   - Link from Custom Models to AutoML (for quick start users)
   - Update all SQL reference examples
   - Update conceptual introduction with decision tree

6. **Version-Specific Documentation**:
   - Clear marking of "New in 2025.2" features
   - Version compatibility matrix
   - Deprecation notices (if any)

**Timeline**: Continuous improvement

## Recommendations for Doc Team Collaboration

### 1. Follow Established Patterns

✅ **DO**:
- Use the same structure as "Using IntegratedML" guide
- Follow SQL syntax documentation patterns
- Use provider architecture (add Custom Models as new provider)
- Maintain separation: User Guide vs Reference Guide
- Keep version-specific documentation approach

❌ **DON'T**:
- Create entirely new documentation structure
- Break from established SQL syntax patterns
- Treat Custom Models as separate from IntegratedML
- Duplicate existing AutoML concepts

### 2. Content Organization

**Propose to Doc Team**:

```
Using IntegratedML Guide (Update Existing)
├── Introduction to IntegratedML (unchanged)
├── IntegratedML Basics (add Custom Models example)
├── ML Configurations (add Custom Models config)
├── Providers (add Custom Models section)
│   ├── AutoML
│   ├── DataRobot
│   ├── H2O
│   ├── PMML
│   └── Custom Models ← NEW
└── When to Use (add decision guidance)

Custom Models Reference Guide (New)
├── Introduction
├── Architecture
├── Base Classes API Reference
├── Creating Custom Models
├── Model Lifecycle
├── Integration with IRIS
├── Performance & Best Practices
└── Examples & Tutorials
```

### 3. Positioning Statement

**Proposed Language for Docs**:

> **Custom Models** extends IntegratedML with the ability to deploy your own Python machine learning models directly within SQL queries. While IntegratedML's AutoML provider automates feature engineering and model selection for rapid development without ML expertise, Custom Models gives data scientists and Python developers full control over model implementation, preprocessing logic, and third-party library integration—all while maintaining IntegratedML's core benefit of in-database execution without data movement.

**When to Use**:
- **Use AutoML** when you need quick models without ML expertise
- **Use Custom Models** when you need custom preprocessing, domain-specific logic, or third-party libraries like Prophet or LightGBM

### 4. Key Messaging Consistency

Maintain consistent messaging with existing docs:

| Concept | Existing IntegratedML | Custom Models Extension |
|---------|----------------------|-------------------------|
| **SQL Interface** | ✅ All SQL | ✅ Same SQL commands |
| **In-Database** | ✅ No data movement | ✅ No data movement |
| **Ease of Use** | ✅ No expertise needed | ⚠️ Python + ML knowledge required |
| **Speed** | ✅ Quick time to value | ⚠️ Development time needed |
| **Control** | ❌ Automated only | ✅ Full control |
| **Flexibility** | ❌ AutoML algorithms | ✅ Any scikit-learn compatible |

## Questions for Doc Team Meeting

### Content Questions

1. **Guide Structure**: Should Custom Models be in "Using IntegratedML" or separate guide?
2. **Versioning**: How to handle "new in 2025.2" markers across versions?
3. **Depth**: How much Python code detail in official docs vs GitHub repo?
4. **Examples**: How many examples in docs vs pointing to GitHub demos?

### Process Questions

5. **Format**: What format for source documentation? (Markdown → DITA conversion?)
6. **Timeline**: What's the deadline for 2026.1 docs freeze?
7. **Review**: What's the technical review process?
8. **Maintenance**: Who owns updates post-launch?

### Integration Questions

9. **Cross-Links**: How to link between docs.intersystems.com and GitHub repo?
10. **Search**: How to ensure Custom Models appears in docs search?
11. **Navigation**: How will users discover Custom Models from AutoML docs?
12. **Migration**: Need migration guide from AutoML to Custom Models?

## Next Steps

1. **Share this analysis** with doc team
2. **Schedule alignment meeting** to review existing docs
3. **Define scope** for 2026.1 documentation
4. **Create documentation plan** based on existing patterns
5. **Identify content sources** (GitHub repo → official docs)
6. **Establish review process** and timeline

## Appendix: Useful Links

### Existing Documentation
- Introduction to IntegratedML: https://docs.intersystems.com/irislatest/csp/docbook/DocBook.UI.Page.cls?KEY=GIML_Intro
- IntegratedML Basics: https://docs.intersystems.com/irislatest/csp/docbook/DocBook.UI.Page.cls?KEY=GIML_Basics
- Providers: https://docs.intersystems.com/irislatest/csp/docbook/DocBook.UI.Page.cls?KEY=GIML_Configuration_Providers
- AutoML Reference: https://docs.intersystems.com/irislatest/csp/docbook/DocBook.UI.Page.cls?KEY=GAUTOML_Intro

### Learning Resources
- IntegratedML Training: https://learning.intersystems.com/course/view.php?name=HandsOnIntegratedML
- Video: What is IntegratedML: https://www.youtube.com/watch?v=1IOF41WgFV4

### GitHub Resources
- IntegratedML Demo Template: https://github.com/intersystems-community/integratedml-demo-template
- Custom Models Repository: https://github.com/intersystems-community/integratedml-custom-models
