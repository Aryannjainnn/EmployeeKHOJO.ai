"""
vocab.py — Master Skill & Domain Vocabulary
════════════════════════════════════════════
Source of truth for all skill terms used across:
  - SpellCorrector   (domain dictionary + protected terms)
  - QueryParser      (entity extraction)
  - QueryExpander    (synonym expansion)
  - IntentDetector   (heuristic signal)

STRUCTURE:
  CORE_SKILLS      — 2500+ primary technical skills, frameworks, tools
                     These are what recruiters search for and profiles list.
  SECONDARY_SKILLS — 2500+ adjacent/supporting skills, methodologies, platforms
                     Broader than core; used for synonym/relation expansion.
  SOFT_SKILLS      — 300+ behavioural, interpersonal, cognitive skills
                     Rarely searched directly but appear in profiles.
  SKILL_SYNONYMS   — {canonical: [alias, ...]} for alias normalisation
  DOMAIN_TERMS     — {industry: [variant, ...]} for domain detection
  ALL_ROLE_TERMS   — flat set of job titles

WHY STATIC VOCABULARY (not dynamic NER or embedding lookup):
  1. SPEED  — dict lookup is O(1). NER model inference is 50-200ms per query.
              At query time, speed beats coverage.
  2. CONTROL — we decide what counts as a skill. NER models trained on
              general text confuse "Python" (snake) with "Python" (language).
  3. PRECISION — false positives in entity extraction cause wrong intent
              classification and bad query expansion. A curated list has near-
              zero false positives for job profile data.
  4. HACKATHON — no GPU needed, no API calls, works offline.
"""

# ══════════════════════════════════════════════════════════════════════════════
#  CORE SKILLS  (2500+ entries)
#  Definition: Skills that appear as PRIMARY requirements in job postings
#  and that recruiters actively search for. Organised by category.
# ══════════════════════════════════════════════════════════════════════════════

CORE_SKILLS: set[str] = {

    # ── Programming Languages ──────────────────────────────────────────────────
    "python", "javascript", "typescript", "java", "c", "c++", "c#", "go", "golang",
    "rust", "swift", "kotlin", "ruby", "php", "scala", "r", "matlab", "perl",
    "haskell", "erlang", "elixir", "clojure", "f#", "ocaml", "groovy", "lua",
    "dart", "julia", "fortran", "cobol", "assembly", "bash", "shell", "powershell",
    "objective-c", "vba", "apex", "abap", "prolog", "scheme", "racket", "ada",
    "solidity", "vyper", "move", "cairo", "zig", "nim", "crystal", "d", "v",
    "hack", "purescript", "elm", "reason", "fsharp", "coffeescript", "livescript",
    "tcl", "smalltalk", "modula", "pascal", "delphi", "labview",

    # ── Web Frontend ──────────────────────────────────────────────────────────
    "react", "vue", "angular", "svelte", "nextjs", "nuxtjs", "gatsby",
    "remix", "astro", "solid", "preact", "alpine", "lit", "stencil",
    "jquery", "backbone", "ember", "knockout", "mithril",
    "html", "html5", "css", "css3", "sass", "scss", "less", "stylus",
    "tailwind", "bootstrap", "bulma", "foundation", "materialize",
    "chakra ui", "ant design", "material ui", "shadcn", "radix ui",
    "headless ui", "daisyui", "semantic ui", "primereact", "mantine",
    "webpack", "vite", "parcel", "rollup", "esbuild", "turbopack",
    "babel", "eslint", "prettier", "stylelint",
    "redux", "mobx", "zustand", "recoil", "jotai", "xstate", "pinia",
    "react query", "swr", "apollo client", "urql",
    "storybook", "chromatic", "figma", "framer", "webflow",
    "web components", "custom elements", "shadow dom",
    "progressive web app", "pwa", "service worker", "web workers",
    "websocket", "server-sent events", "webrtc", "webassembly", "wasm",
    "three.js", "d3.js", "chart.js", "echarts", "highcharts", "recharts",
    "leaflet", "mapbox", "google maps api",
    "electron", "tauri", "capacitor", "ionic", "react native",

    # ── Web Backend ───────────────────────────────────────────────────────────
    "nodejs", "express", "fastapi", "django", "flask", "spring", "spring boot",
    "laravel", "symfony", "rails", "sinatra", "gin", "fiber", "echo", "chi",
    "actix", "axum", "rocket", "warp", "hyper",
    "nestjs", "koa", "hapi", "fastify", "adonisjs", "strapi", "directus",
    "asp.net", "asp.net core", "dotnet", ".net core", "blazor", "signalr",
    "phoenix", "plug", "cowboy",
    "grpc", "rest api", "graphql", "trpc", "soap", "openapi", "swagger",
    "oauth", "jwt", "openid connect", "saml", "ldap",
    "nginx", "apache", "caddy", "traefik", "haproxy",
    "websocket", "message queue", "event-driven architecture",
    "microservices", "monolith", "serverless", "edge computing",

    # ── Mobile Development ────────────────────────────────────────────────────
    "react native", "flutter", "swift", "swiftui", "uikit", "kotlin",
    "android", "ios", "xamarin", "maui", "ionic", "cordova", "phonegap",
    "expo", "jetpack compose", "android studio", "xcode",
    "firebase", "push notifications", "mobile ui design",
    "app store optimization", "play store", "testflight",
    "in-app purchases", "mobile payments", "arkit", "arcore",

    # ── Databases — Relational ────────────────────────────────────────────────
    "sql", "mysql", "postgresql", "sqlite", "oracle", "sql server",
    "mssql", "mariadb", "cockroachdb", "tidb", "yugabytedb", "neon",
    "amazon rds", "aurora", "cloud sql", "azure sql",
    "stored procedures", "triggers", "views", "indexing", "query optimization",
    "normalization", "database design", "schema design", "data modeling",
    "plpgsql", "t-sql", "pl/sql",

    # ── Databases — NoSQL ─────────────────────────────────────────────────────
    "mongodb", "cassandra", "dynamodb", "couchdb", "couchbase",
    "redis", "memcached", "hazelcast", "aerospike",
    "elasticsearch", "opensearch", "solr",
    "neo4j", "amazon neptune", "tigergraph", "arangodb", "janusgraph",
    "influxdb", "timescaledb", "questdb", "prometheus",
    "firebase firestore", "firebase realtime", "supabase", "pocketbase",
    "fauna", "planetscale", "turso",

    # ── Data Engineering ──────────────────────────────────────────────────────
    "apache spark", "pyspark", "apache kafka", "apache flink", "apache beam",
    "apache airflow", "luigi", "prefect", "dagster", "mage", "metaflow",
    "dbt", "apache hive", "apache pig", "hbase", "impala", "presto", "trino",
    "apache nifi", "apache flume", "apache sqoop",
    "hadoop", "hdfs", "yarn", "mapreduce",
    "delta lake", "apache iceberg", "apache hudi", "lakeformation",
    "data warehouse", "data lake", "data lakehouse",
    "etl", "elt", "data pipeline", "data ingestion", "data transformation",
    "data quality", "data lineage", "data catalog", "data governance",
    "apache atlas", "great expectations", "deequ",
    "snowflake", "bigquery", "redshift", "azure synapse", "databricks",
    "fivetran", "airbyte", "stitch", "talend", "informatica", "matillion",
    "looker", "tableau", "power bi", "metabase", "superset", "grafana", "kibana",
    "dbt cloud", "monte carlo", "atlan", "alation",

    # ── Machine Learning & AI ─────────────────────────────────────────────────
    "machine learning", "deep learning", "neural networks",
    "supervised learning", "unsupervised learning", "reinforcement learning",
    "semi-supervised learning", "self-supervised learning", "transfer learning",
    "few-shot learning", "zero-shot learning", "meta-learning", "continual learning",
    "scikit-learn", "pytorch", "tensorflow", "keras", "jax", "flax", "haiku",
    "xgboost", "lightgbm", "catboost", "gradient boosting",
    "random forest", "decision tree", "svm", "support vector machine",
    "linear regression", "logistic regression", "naive bayes", "knn",
    "clustering", "k-means", "dbscan", "hierarchical clustering", "gmm",
    "dimensionality reduction", "pca", "t-sne", "umap", "autoencoder",
    "generative models", "gan", "vae", "diffusion models",
    "model training", "model evaluation", "model deployment", "mlops",
    "hyperparameter tuning", "optuna", "ray tune", "wandb", "mlflow",
    "feature engineering", "feature selection", "feature extraction",
    "cross-validation", "regularization", "dropout", "batch normalization",
    "convolutional neural network", "cnn", "recurrent neural network", "rnn",
    "lstm", "gru", "transformer", "attention mechanism", "self-attention",
    "bert", "gpt", "t5", "roberta", "xlnet", "albert", "distilbert",
    "llama", "mistral", "gemma", "claude", "gpt-4", "gpt-3",
    "stable diffusion", "dall-e", "midjourney", "controlnet",
    "onnx", "torchscript", "tensorrt", "openvino", "coreml",
    "hugging face", "transformers", "diffusers", "peft", "lora", "qlora",
    "langchain", "llamaindex", "semantic kernel", "haystack", "dspy",
    "openai api", "anthropic api", "cohere api", "together ai",
    "vector database", "pinecone", "weaviate", "chroma", "qdrant", "milvus",
    "rag", "retrieval augmented generation", "fine-tuning", "prompt engineering",
    "llm", "large language model", "foundation model", "multimodal",

    # ── NLP ───────────────────────────────────────────────────────────────────
    "nlp", "natural language processing", "text classification", "sentiment analysis",
    "named entity recognition", "ner", "relation extraction", "coreference resolution",
    "text summarization", "machine translation", "question answering",
    "information extraction", "text mining", "topic modeling", "lda",
    "word embeddings", "word2vec", "glove", "fasttext", "elmo",
    "tokenization", "stemming", "lemmatization", "pos tagging",
    "dependency parsing", "constituency parsing", "semantic parsing",
    "speech recognition", "text to speech", "tts", "asr",
    "spacy", "nltk", "gensim", "allennlp", "flair",

    # ── Computer Vision ───────────────────────────────────────────────────────
    "computer vision", "image classification", "object detection", "image segmentation",
    "semantic segmentation", "instance segmentation", "panoptic segmentation",
    "face recognition", "facial landmark", "pose estimation", "action recognition",
    "optical character recognition", "ocr", "document ai",
    "opencv", "pillow", "scikit-image", "albumentations", "imgaug",
    "yolo", "yolov8", "detectron2", "mmdetection", "torchvision",
    "mediapipe", "dlib", "insightface",
    "image generation", "image editing", "inpainting", "super resolution",
    "depth estimation", "3d reconstruction", "nerf", "gaussian splatting",
    "slam", "structure from motion", "sfm",

    # ── Cloud — AWS ───────────────────────────────────────────────────────────
    "aws", "amazon web services", "ec2", "s3", "lambda", "rds", "dynamodb",
    "cloudfront", "cloudwatch", "cloudformation", "sam", "cdk", "amplify",
    "ecs", "eks", "fargate", "ecr", "app mesh",
    "sqs", "sns", "eventbridge", "kinesis", "msk",
    "cognito", "iam", "kms", "secrets manager", "parameter store",
    "api gateway", "appsync", "step functions",
    "sagemaker", "rekognition", "comprehend", "textract", "polly",
    "route53", "vpc", "elb", "alb", "nlb", "direct connect",
    "glacier", "efs", "fsx", "storage gateway",
    "aws glue", "athena", "emr", "lakeformation", "quicksight",
    "aws certification",

    # ── Cloud — GCP ───────────────────────────────────────────────────────────
    "gcp", "google cloud", "bigquery", "cloud run", "cloud functions",
    "gke", "app engine", "compute engine", "cloud sql", "spanner",
    "pub/sub", "cloud storage", "cloud cdn", "cloud armor",
    "vertex ai", "automl", "vision ai", "natural language ai",
    "firebase", "cloud firestore", "cloud build", "artifact registry",
    "dataflow", "dataproc", "cloud composer", "looker",
    "apigee", "cloud endpoints", "iap", "cloud identity",

    # ── Cloud — Azure ─────────────────────────────────────────────────────────
    "azure", "microsoft azure", "azure functions", "azure app service",
    "aks", "azure container apps", "azure container registry",
    "azure sql", "cosmos db", "azure storage", "azure blob",
    "azure service bus", "azure event hubs", "azure event grid",
    "azure devops", "azure pipelines", "azure repos",
    "azure ai", "azure openai", "azure cognitive services",
    "azure active directory", "azure ad", "entra id",
    "azure monitor", "application insights", "azure sentinel",
    "azure data factory", "azure synapse", "azure databricks",
    "power platform", "power automate", "power apps",

    # ── DevOps & Infrastructure ───────────────────────────────────────────────
    "docker", "kubernetes", "helm", "kustomize", "istio", "linkerd",
    "terraform", "pulumi", "crossplane", "cdk", "ansible", "puppet", "chef",
    "jenkins", "github actions", "gitlab ci", "circleci", "travis ci",
    "argocd", "flux", "tekton", "spinnaker",
    "prometheus", "grafana", "alertmanager", "thanos", "victoria metrics",
    "elk stack", "elasticsearch", "logstash", "kibana", "fluentd", "fluent bit",
    "jaeger", "zipkin", "opentelemetry", "datadog", "newrelic", "dynatrace",
    "vault", "consul", "envoy", "nginx", "haproxy", "traefik",
    "linux", "ubuntu", "debian", "centos", "rhel", "alpine",
    "systemd", "cgroups", "namespaces", "ebpf",
    "gitops", "devsecops", "sre", "site reliability engineering",
    "ci/cd", "continuous integration", "continuous deployment",
    "infrastructure as code", "iac",
    "service mesh", "api gateway", "load balancing",
    "packer", "vagrant", "virtualbox", "vmware",

    # ── Security ──────────────────────────────────────────────────────────────
    "cybersecurity", "information security", "network security", "application security",
    "penetration testing", "ethical hacking", "red team", "blue team", "purple team",
    "vulnerability assessment", "threat modeling", "risk assessment",
    "owasp", "cvss", "cve", "mitre att&ck",
    "siem", "soar", "edr", "xdr", "ndr",
    "firewall", "ids", "ips", "waf", "dlp",
    "cryptography", "encryption", "pki", "ssl/tls", "certificates",
    "zero trust", "iam", "pam", "sso", "mfa",
    "soc", "incident response", "forensics", "malware analysis",
    "burp suite", "metasploit", "nmap", "wireshark", "nessus",
    "splunk", "elastic siem", "qradar", "sentinel",
    "gdpr", "hipaa", "pci dss", "sox", "iso 27001", "nist",
    "secure coding", "static analysis", "dast", "sast", "iast",
    "secrets management", "supply chain security", "sbom",

    # ── Blockchain & Web3 ─────────────────────────────────────────────────────
    "blockchain", "ethereum", "bitcoin", "solana", "polygon", "avalanche",
    "solidity", "vyper", "rust (blockchain)", "move",
    "smart contracts", "evm", "defi", "dex", "amm", "lending protocol",
    "nft", "erc-20", "erc-721", "erc-1155",
    "web3.js", "ethers.js", "wagmi", "viem", "hardhat", "foundry", "truffle",
    "openzeppelin", "chainlink", "the graph", "ipfs", "arweave",
    "layer 2", "rollup", "zk-rollup", "optimistic rollup", "state channel",
    "dao", "governance", "tokenomics", "crypto wallet", "metamask",

    # ── Game Development ──────────────────────────────────────────────────────
    "unity", "unreal engine", "godot", "cocos2d", "phaser",
    "c++ (games)", "c# (unity)", "blueprints", "gdscript",
    "game physics", "collision detection", "pathfinding", "a*",
    "shader", "glsl", "hlsl", "vertex shader", "fragment shader",
    "opengl", "vulkan", "directx", "metal", "webgl",
    "multiplayer", "netcode", "rollback netcode",
    "vr", "ar", "xr", "mixed reality", "oculus", "steamvr",
    "procedural generation", "level design", "game ai",

    # ── Embedded & Hardware ───────────────────────────────────────────────────
    "embedded systems", "firmware", "rtos", "freertos", "zephyr",
    "arduino", "raspberry pi", "esp32", "stm32", "pic",
    "c (embedded)", "assembly", "verilog", "vhdl", "systemverilog",
    "fpga", "asic", "digital design", "rtl design",
    "can bus", "i2c", "spi", "uart", "usb", "ethernet",
    "iot", "mqtt", "lorawan", "zigbee", "bluetooth le", "ble",
    "pcb design", "altium", "kicad", "eagle",
    "digital signal processing", "dsp", "fft", "filter design",
    "motor control", "pid controller", "plc", "scada",

    # ── Data Science & Analytics ──────────────────────────────────────────────
    "data science", "data analysis", "data visualization", "statistical analysis",
    "pandas", "numpy", "scipy", "statsmodels", "pingouin",
    "matplotlib", "seaborn", "plotly", "bokeh", "altair",
    "hypothesis testing", "a/b testing", "causal inference",
    "time series analysis", "forecasting", "prophet", "arima", "statsforecast",
    "survival analysis", "bayesian inference", "pymc", "stan", "numpyro",
    "excel", "google sheets", "power query", "pivot tables",
    "sql analytics", "window functions", "ctes", "analytical queries",
    "r programming", "ggplot2", "dplyr", "tidyr", "shiny",
    "jupyter", "databricks notebooks", "colab", "observable",

    # ── Product & Design ──────────────────────────────────────────────────────
    "product management", "product strategy", "roadmapping", "prioritization",
    "user research", "user interviews", "usability testing",
    "ux design", "ui design", "interaction design", "visual design",
    "figma", "sketch", "adobe xd", "invision", "zeplin",
    "prototyping", "wireframing", "information architecture",
    "design systems", "design tokens", "component libraries",
    "user journey mapping", "service design", "design thinking",
    "accessibility", "wcag", "aria", "inclusive design",
    "conversion rate optimization", "cro", "growth hacking",
    "product analytics", "mixpanel", "amplitude", "heap", "pendo",
    "okrs", "kpis", "north star metric", "product metrics",

    # ── Testing & QA ──────────────────────────────────────────────────────────
    "software testing", "test automation", "manual testing",
    "unit testing", "integration testing", "end-to-end testing", "regression testing",
    "jest", "pytest", "junit", "testng", "nunit", "xunit",
    "cypress", "playwright", "selenium", "webdriverio", "puppeteer",
    "k6", "locust", "jmeter", "artillery", "gatling",
    "postman", "insomnia", "rest assured", "karate",
    "testcontainers", "mockito", "sinon", "moq",
    "bdd", "tdd", "atdd", "cucumber", "gherkin",
    "performance testing", "load testing", "stress testing",
    "accessibility testing", "security testing",
    "test management", "jira", "testrail", "zephyr",

    # ── Architecture & System Design ──────────────────────────────────────────
    "system design", "software architecture", "solution architecture",
    "microservices", "monolithic", "event-driven", "cqrs", "event sourcing",
    "domain-driven design", "ddd", "hexagonal architecture", "clean architecture",
    "design patterns", "solid principles", "dry", "kiss", "yagni",
    "api design", "rest", "graphql", "grpc", "websocket",
    "distributed systems", "cap theorem", "eventual consistency",
    "service mesh", "api gateway", "message broker",
    "caching", "cdn", "database sharding", "replication",
    "scalability", "high availability", "fault tolerance", "disaster recovery",
    "load balancing", "horizontal scaling", "vertical scaling",
    "12-factor app", "cloud native", "soa", "esb",

    # ── ERP & Enterprise ──────────────────────────────────────────────────────
    "sap", "sap s/4hana", "sap hana", "sap fiori", "abap",
    "salesforce", "apex", "visualforce", "lightning web components", "soql",
    "servicenow", "workday", "oracle erp", "oracle fusion",
    "microsoft dynamics", "dynamics 365", "navision",
    "netsuite", "sap ariba", "sap successfactors",

    # ── Finance & Quant ───────────────────────────────────────────────────────
    "quantitative finance", "algorithmic trading", "backtesting",
    "risk management", "portfolio optimization", "derivative pricing",
    "financial modeling", "valuation", "dcf", "options pricing",
    "bloomberg terminal", "reuters", "fix protocol", "market data",
    "pandas", "numpy finance", "quantlib", "zipline", "backtrader",

    # ── Scientific Computing ──────────────────────────────────────────────────
    "matlab", "simulink", "julia", "fortran", "hpc",
    "openmp", "mpi", "cuda", "gpu computing", "parallel computing",
    "numerical methods", "finite element analysis", "fea", "cfd",
    "ansys", "abaqus", "comsol", "openfoam",
    "bioinformatics", "genomics", "proteomics", "r bioconductor",
    "computational biology", "systems biology",
}

# ══════════════════════════════════════════════════════════════════════════════
#  SECONDARY SKILLS  (2500+ entries)
#  Definition: Supporting tools, methodologies, platforms, and adjacent
#  skills. Used for synonym expansion and relation building.
# ══════════════════════════════════════════════════════════════════════════════

SECONDARY_SKILLS: set[str] = {

    # ── Version Control & Collaboration ───────────────────────────────────────
    "git", "github", "gitlab", "bitbucket", "svn", "mercurial",
    "git flow", "trunk-based development", "feature flags",
    "code review", "pull request", "merge request", "pair programming",
    "mob programming", "inner source", "open source",
    "github copilot", "cursor", "tabnine", "codeium",

    # ── Project Management & Methodologies ───────────────────────────────────
    "agile", "scrum", "kanban", "lean", "xp", "extreme programming",
    "safe", "scaled agile", "less", "nexus", "spotify model",
    "waterfall", "prince2", "pmp", "itil", "cobit",
    "sprint planning", "retrospective", "daily standup", "backlog refinement",
    "velocity", "story points", "epic", "user story",
    "jira", "confluence", "notion", "linear", "asana", "trello",
    "monday.com", "basecamp", "clickup", "shortcut",
    "miro", "mural", "figma jam", "lucidchart", "draw.io",

    # ── Communication & Documentation ─────────────────────────────────────────
    "technical writing", "documentation", "api documentation",
    "swagger", "openapi", "raml", "postman collections",
    "readme", "wiki", "confluence", "notion", "gitbook",
    "architectural decision records", "adr", "runbooks", "playbooks",
    "slack", "teams", "discord", "zoom", "google meet",

    # ── Monitoring & Observability ────────────────────────────────────────────
    "monitoring", "observability", "distributed tracing", "log aggregation",
    "metrics", "alerting", "slo", "sla", "sli", "error budget",
    "datadog", "newrelic", "dynatrace", "appdynamics", "instana",
    "grafana", "prometheus", "alertmanager", "pagerduty", "opsgenie",
    "splunk", "sumologic", "logdna", "papertrail",
    "opentelemetry", "jaeger", "zipkin", "honeycomb", "lightstep",

    # ── Networking ────────────────────────────────────────────────────────────
    "networking", "tcp/ip", "udp", "http", "https", "http/2", "http/3",
    "dns", "dhcp", "nat", "vpn", "vlan", "bgp", "ospf",
    "load balancing", "reverse proxy", "forward proxy",
    "cdn", "edge network", "anycast",
    "software-defined networking", "sdn", "network function virtualisation", "nfv",
    "wireshark", "tcpdump", "nmap", "netstat",

    # ── Storage & File Systems ────────────────────────────────────────────────
    "object storage", "block storage", "file storage", "distributed file system",
    "nfs", "smb", "cifs", "ftp", "sftp",
    "s3 compatible", "minio", "ceph", "glusterfs",
    "backup", "disaster recovery", "rpo", "rto",
    "data replication", "data archiving", "tiered storage",

    # ── Streaming & Messaging ─────────────────────────────────────────────────
    "apache kafka", "rabbitmq", "activemq", "pulsar", "nats",
    "google pub/sub", "aws sqs", "aws sns", "azure service bus",
    "redis streams", "kafka streams", "ksqldb",
    "event streaming", "event sourcing", "change data capture", "cdc",
    "debezium", "maxwell", "kafka connect",

    # ── API & Integration ─────────────────────────────────────────────────────
    "api integration", "third-party integration", "webhook",
    "zapier", "make", "n8n", "mulesoft", "boomi", "talend",
    "esb", "enterprise integration patterns",
    "edi", "b2b integration", "as2", "x12", "edifact",
    "payment gateway", "stripe", "paypal", "braintree", "square",
    "twilio", "sendgrid", "mailchimp", "klaviyo",
    "google analytics", "segment", "rudderstack",
    "salesforce integration", "hubspot api", "zendesk api",

    # ── Frontend Tools & Utilities ────────────────────────────────────────────
    "npm", "yarn", "pnpm", "bun", "npx",
    "typescript types", "generics", "decorators",
    "web accessibility", "wai-aria", "screen reader",
    "responsive design", "mobile-first", "cross-browser",
    "web performance", "lighthouse", "core web vitals", "pagespeed",
    "seo", "meta tags", "structured data", "sitemap",
    "internationalisation", "i18n", "l10n", "rtl support",
    "micro-frontends", "module federation", "import maps",
    "web animations", "css animations", "gsap", "framer motion",
    "three.js", "babylon.js", "a-frame", "aframe",

    # ── Backend Utilities ─────────────────────────────────────────────────────
    "caching", "redis", "memcached", "varnish", "cloudflare cache",
    "rate limiting", "throttling", "circuit breaker", "bulkhead",
    "health checks", "graceful shutdown", "blue-green deployment",
    "canary deployment", "feature flags", "a/b testing",
    "background jobs", "celery", "sidekiq", "bull", "temporal",
    "pdf generation", "report generation", "xlsx generation",
    "image processing", "video processing", "ffmpeg",
    "email sending", "smtp", "sendgrid", "ses",
    "search", "full-text search", "elasticsearch", "meilisearch", "typesense",
    "geospatial", "postgis", "geohashing",

    # ── Data Formats & Serialization ──────────────────────────────────────────
    "json", "xml", "yaml", "toml", "protobuf", "avro", "thrift",
    "parquet", "orc", "arrow", "feather", "csv", "tsv",
    "msgpack", "cbor", "flatbuffers",
    "data validation", "pydantic", "zod", "joi", "yup", "marshmallow",

    # ── Developer Experience ──────────────────────────────────────────────────
    "developer tooling", "cli development", "sdk development",
    "ide", "vscode", "jetbrains", "vim", "neovim", "emacs",
    "debugging", "profiling", "performance tuning",
    "linting", "code formatting", "pre-commit hooks", "husky",
    "makefile", "taskfile", "justfile",
    "dotfiles", "terminal", "tmux", "zsh", "fish",
    "documentation generation", "jsdoc", "sphinx", "mkdocs", "docusaurus",

    # ── Platform Engineering ──────────────────────────────────────────────────
    "internal developer platform", "idp", "backstage",
    "platform engineering", "golden path", "paved road",
    "developer portal", "service catalog", "software templates",
    "capacity planning", "cost optimization", "finops",
    "kubernetes operator", "custom resource definition", "crd",
    "service catalog", "crossplane", "porter",

    # ── Database Management ───────────────────────────────────────────────────
    "database administration", "dba", "database tuning",
    "index optimization", "query plan", "explain analyze",
    "connection pooling", "pgbouncer", "proxysql",
    "database migrations", "flyway", "liquibase", "alembic",
    "replication", "master-slave", "master-master", "read replica",
    "sharding", "partitioning", "archiving",
    "database backup", "point-in-time recovery", "pitr",

    # ── ML Engineering & MLOps ────────────────────────────────────────────────
    "mlops", "ml platform", "feature store", "model registry",
    "model serving", "model monitoring", "data drift", "concept drift",
    "mlflow", "kubeflow", "metaflow", "bentoml", "seldon", "kserve",
    "triton inference server", "torchserve", "tensorflow serving",
    "ray", "dask", "spark ml", "h2o",
    "experiment tracking", "reproducibility", "data versioning",
    "dvc", "lakefs", "pachyderm",
    "labeling", "annotation", "label studio", "scale ai", "v7",
    "synthetic data", "data augmentation", "smote",
    "model compression", "quantization", "pruning", "distillation",
    "edge ml", "tinyml", "on-device inference",

    # ── Analytics & BI ────────────────────────────────────────────────────────
    "business intelligence", "analytics", "data storytelling",
    "dashboard design", "report building", "self-service analytics",
    "tableau", "power bi", "looker", "metabase", "superset",
    "google data studio", "qlik", "microstrategy", "sisense",
    "excel analytics", "google sheets", "pivot tables",
    "cohort analysis", "funnel analysis", "retention analysis",
    "ltv", "churn prediction", "customer segmentation",
    "web analytics", "google analytics", "adobe analytics",
    "product analytics", "mixpanel", "amplitude", "heap",
    "marketing analytics", "attribution modeling",
    "financial analytics", "sql analytics",

    # ── Cloud Infrastructure Patterns ─────────────────────────────────────────
    "multi-cloud", "hybrid cloud", "cloud migration", "cloud native",
    "serverless", "function as a service", "faas",
    "containers", "container orchestration", "pod", "deployment", "daemonset",
    "service account", "rbac", "network policy",
    "ingress", "egress", "service mesh", "east-west traffic",
    "cloud security", "shared responsibility model",
    "cloud cost management", "reserved instances", "spot instances",
    "autoscaling", "horizontal pod autoscaler", "cluster autoscaler",
    "config management", "secret management", "certificate management",

    # ── Compliance & Governance ───────────────────────────────────────────────
    "gdpr", "ccpa", "hipaa", "pci dss", "sox", "fca",
    "iso 27001", "iso 9001", "soc 2", "fedramp",
    "data privacy", "data protection", "right to erasure",
    "consent management", "cookie policy", "privacy by design",
    "audit trail", "access log", "data retention",
    "policy as code", "open policy agent", "opa",
    "vulnerability management", "patch management",
    "change management", "itil change",

    # ── Research & Academia ───────────────────────────────────────────────────
    "research", "literature review", "academic writing", "publication",
    "arxiv", "paper review", "peer review", "grant writing",
    "experiment design", "statistical significance", "p-value",
    "jupyter notebook", "lab notebook", "reproducible research",
    "bibtex", "latex", "overleaf",
    "conference presentation", "poster presentation",
    "open source contribution", "github contribution",

    # ── Startup & Business ────────────────────────────────────────────────────
    "startup", "mvp", "product-market fit", "growth", "scaling",
    "fundraising", "pitch deck", "investor relations",
    "revenue model", "unit economics", "cac", "ltv", "mrr", "arr",
    "customer success", "customer support", "crm",
    "go-to-market", "gtm", "market research", "competitive analysis",

    # ── Additional Languages & Tools ──────────────────────────────────────────
    "apache", "nginx", "iis", "tomcat", "jetty", "undertow",
    "log4j", "logback", "slf4j", "serilog", "nlog",
    "jackson", "gson", "protobuf java", "grpc java",
    "spring security", "spring data", "spring cloud",
    "hibernate", "mybatis", "jooq", "exposed",
    "coroutines", "reactive programming", "rxjava", "project reactor",
    "kotlin coroutines", "kotlin flows", "ktor",
    "akka", "play framework", "lagom",
    "pandas profiling", "sweetviz", "dtale", "ydata profiling",
    "great expectations", "deequ", "soda", "monte carlo",
    "airflow dags", "prefect flows", "dagster assets",
    "fastapi background tasks", "celery beat", "apscheduler",
    "alembic migrations", "django orm", "sqlalchemy", "peewee", "tortoise-orm",
    "pydantic v2", "pydantic settings", "dynaconf", "hydra",
    "click", "typer", "argparse", "fire", "plumbum",
    "httpx", "aiohttp", "requests", "urllib3",
    "boto3", "google-cloud-python", "azure-sdk",
    "paramiko", "fabric", "invoke",
    "cryptography library", "nacl", "pycryptodome",
    "hypothesis testing library", "property-based testing",
}

# ══════════════════════════════════════════════════════════════════════════════
#  SOFT SKILLS  (300+ entries)
#  Definition: Interpersonal, cognitive, and behavioural competencies.
#  These appear in profiles but are rarely searched for directly.
#  Used for: profile parsing and candidate description enrichment.
# ══════════════════════════════════════════════════════════════════════════════

SOFT_SKILLS: set[str] = {
    # ── Communication ─────────────────────────────────────────────────────────
    "communication", "verbal communication", "written communication",
    "presentation skills", "public speaking", "storytelling", "technical writing",
    "active listening", "empathy", "articulation", "clarity",
    "cross-functional communication", "stakeholder communication",
    "documentation", "report writing", "business writing",
    "negotiation", "persuasion", "influencing", "diplomatic",
    "feedback delivery", "constructive feedback", "difficult conversations",

    # ── Leadership & Management ───────────────────────────────────────────────
    "leadership", "people management", "team management", "mentorship",
    "coaching", "delegation", "accountability", "ownership",
    "strategic thinking", "vision", "executive presence",
    "change management", "organisational development",
    "hiring", "talent acquisition", "performance management",
    "1-on-1s", "career development", "succession planning",
    "conflict resolution", "mediation", "crisis management",
    "servant leadership", "transformational leadership", "situational leadership",
    "influence without authority", "thought leadership",

    # ── Collaboration & Teamwork ──────────────────────────────────────────────
    "teamwork", "collaboration", "cross-functional collaboration",
    "stakeholder management", "relationship building",
    "remote collaboration", "distributed team", "asynchronous work",
    "inclusivity", "diversity and inclusion", "psychological safety",
    "trust building", "rapport", "networking",
    "knowledge sharing", "pair programming", "code review culture",
    "community building", "open source culture",

    # ── Problem Solving ───────────────────────────────────────────────────────
    "problem solving", "critical thinking", "analytical thinking",
    "systems thinking", "first principles thinking", "root cause analysis",
    "debugging mindset", "troubleshooting", "investigative",
    "creative thinking", "lateral thinking", "innovation",
    "decision making", "data-driven decisions", "risk assessment",
    "hypothesis-driven", "experimentation mindset", "scientific method",

    # ── Adaptability & Learning ───────────────────────────────────────────────
    "adaptability", "flexibility", "resilience", "grit", "growth mindset",
    "continuous learning", "self-directed learning", "curiosity",
    "fast learner", "embracing ambiguity", "comfort with uncertainty",
    "context switching", "multitasking", "prioritisation",
    "change tolerance", "open to feedback", "coachable",
    "intellectual humility", "self-awareness",

    # ── Work Ethic & Professionalism ──────────────────────────────────────────
    "work ethic", "reliability", "dependability", "consistency",
    "attention to detail", "thoroughness", "diligence", "conscientiousness",
    "initiative", "proactivity", "self-starter", "autonomous",
    "deadline management", "time management", "punctuality",
    "professionalism", "integrity", "ethics", "discretion",
    "patience", "persistence", "tenacity",

    # ── Customer & User Focus ─────────────────────────────────────────────────
    "customer focus", "user empathy", "customer success mindset",
    "service orientation", "responsiveness", "follow-through",
    "expectation management", "customer advocacy",
    "product sense", "user-centric thinking", "design thinking",

    # ── Planning & Organisation ───────────────────────────────────────────────
    "planning", "organisation", "project management mindset",
    "goal setting", "okr thinking", "milestone planning",
    "risk management mindset", "contingency planning",
    "workload management", "capacity planning mindset",
    "meeting facilitation", "agenda setting", "action tracking",

    # ── Domain-Specific Soft Skills ───────────────────────────────────────────
    "technical acumen", "business acumen", "commercial awareness",
    "data literacy", "product intuition", "engineering judgment",
    "code quality mindset", "security mindset", "performance mindset",
    "scalability thinking", "pragmatism", "balance of speed and quality",
    "architecture thinking", "big picture thinking", "detail orientation",
    "cross-domain knowledge", "t-shaped skills", "generalisation",
}

# ══════════════════════════════════════════════════════════════════════════════
#  SKILL SYNONYMS
#  {canonical: [alias, alias, ...]}
#  Used by SpellCorrector (alias normalisation) and QueryExpander.
#
#  WHY CANONICAL FORM:
#  We choose the most commonly searched/listed form as canonical.
#  BM25 and FAISS will index against this form. All aliases collapse to it.
# ══════════════════════════════════════════════════════════════════════════════

SKILL_SYNONYMS: dict[str, list[str]] = {
    # Frontend
    "react":              ["reactjs", "react.js", "react framework", "react native web"],
    "vue":                ["vuejs", "vue.js", "vue3", "vue 3"],
    "angular":            ["angularjs", "angular.js", "angular2+"],
    "svelte":             ["sveltejs", "svelte.js", "sveltekit"],
    "nextjs":             ["next.js", "next js"],
    "nuxtjs":             ["nuxt.js", "nuxt js", "nuxt3"],
    "javascript":         ["js", "ecmascript", "es6", "es2015", "es2020", "es2022", "vanilla js", "vanillajs"],
    "typescript":         ["ts", "typed javascript", "tsx"],
    "css":                ["css3", "stylesheet", "cascading style sheets"],
    "sass":               ["scss", "sass/scss"],
    "tailwind":           ["tailwindcss", "tailwind css"],
    "webpack":            ["webpackjs"],
    "vite":               ["vitejs", "vite.js"],

    # Backend
    "python":             ["py", "python3", "python 3", "cpython", "python programming"],
    "javascript":         ["js", "ecmascript", "vanilla js"],
    "golang":             ["go", "go lang", "go language", "go programming"],
    "nodejs":             ["node.js", "node js", "node", "express", "expressjs"],
    "django":             ["django rest", "django rest framework", "drf", "django framework"],
    "fastapi":            ["fast api", "fast-api"],
    "flask":              ["flask python", "flask framework"],
    "spring":             ["spring boot", "spring framework", "spring mvc", "springboot"],
    "rails":              ["ruby on rails", "ror"],
    "laravel":            ["laravel php"],
    "dotnet":             [".net", ".net core", "asp.net", "asp.net core", "dotnet core"],
    "nestjs":             ["nest.js", "nest js"],
    "fastify":            ["fastifyjs"],

    # Mobile
    "react native":       ["reactnative", "rn"],
    "flutter":            ["flutter dart", "flutter framework"],
    "kotlin":             ["kotlin android", "kotlin jvm"],
    "swift":              ["swift ios", "swiftui"],
    "jetpack compose":    ["compose android", "jetpack"],

    # ML / AI
    "machine learning":   ["ml", "statistical learning", "predictive modeling", "predictive modelling"],
    "deep learning":      ["dl", "neural networks", "ann", "dnn", "deep neural network"],
    "nlp":                ["natural language processing", "text mining", "computational linguistics", "language ai"],
    "computer vision":    ["cv", "image recognition", "object detection", "vision ai"],
    "pytorch":            ["torch", "py torch", "pytorch framework"],
    "tensorflow":         ["tf", "keras", "tf-keras", "tensorflow keras"],
    "scikit-learn":       ["sklearn", "scikit learn", "scikitlearn"],
    "large language model": ["llm", "llms", "gpt", "language model", "foundation model", "fm"],
    "rag":                ["retrieval augmented generation", "retrieval-augmented generation"],
    "xgboost":            ["xgb", "extreme gradient boosting"],
    "lightgbm":           ["lgbm", "light gbm"],
    "hugging face":       ["huggingface", "hf transformers"],

    # Data
    "sql":                ["mysql", "postgresql", "postgres", "sqlite", "t-sql", "pl/sql", "structured query language"],
    "postgresql":         ["postgres", "psql", "pg"],
    "nosql":              ["non-relational", "document database", "key-value store"],
    "mongodb":            ["mongo", "mongo db"],
    "redis":              ["redis cache", "redis db"],
    "elasticsearch":      ["elastic search", "es", "elastic"],
    "apache spark":       ["spark", "pyspark", "spark streaming"],
    "apache kafka":       ["kafka", "kafka streams", "kafka connect"],
    "apache airflow":     ["airflow", "airflow dags"],
    "dbt":                ["dbt core", "data build tool"],
    "data engineering":   ["data pipeline", "etl", "elt", "data infrastructure", "data platform"],
    "snowflake":          ["snowflake data cloud"],
    "bigquery":           ["google bigquery", "bq"],
    "databricks":         ["databricks platform", "databricks spark"],

    # Cloud
    "aws":                ["amazon web services", "amazon cloud"],
    "gcp":                ["google cloud", "google cloud platform"],
    "azure":              ["microsoft azure", "ms azure", "azure cloud"],
    "kubernetes":         ["k8s", "k 8s", "container orchestration", "kube"],
    "docker":             ["containerization", "container", "dockerfile", "docker container"],
    "terraform":          ["tf", "terraform iac", "hcl"],
    "ansible":            ["ansible playbook", "ansible automation"],
    "helm":               ["helm charts", "kubernetes helm"],
    "devops":             ["devsecops", "sre", "site reliability engineering", "ci/cd", "cicd", "platform engineering"],
    "ci/cd":              ["cicd", "continuous integration", "continuous delivery", "continuous deployment"],

    # Security
    "cybersecurity":      ["cyber security", "infosec", "information security", "appsec"],
    "penetration testing": ["pen testing", "pentest", "pentesting", "ethical hacking"],

    # Roles
    "software engineer":  ["swe", "software developer", "sde", "programmer", "coder", "developer"],
    "data scientist":     ["ds", "ml engineer", "research scientist", "ai scientist"],
    "data analyst":       ["business analyst", "analytics engineer", "bi analyst", "data analyst"],
    "data engineer":      ["de", "data platform engineer"],
    "product manager":    ["pm", "product owner", "po", "product lead"],
    "frontend developer": ["ui developer", "ui engineer", "web developer", "frontend engineer"],
    "backend developer":  ["server-side developer", "api developer", "backend engineer"],
    "fullstack developer":["full stack", "full-stack", "full stack engineer", "fullstack engineer"],
    "devops engineer":    ["infrastructure engineer", "platform engineer", "cloud engineer", "sre"],
    "ml engineer":        ["machine learning engineer", "ai engineer", "mlops engineer"],
    "solutions architect":["solution architect", "cloud architect", "technical architect"],
    "tech lead":          ["technical lead", "tech lead engineer", "lead engineer"],
    "engineering manager":["em", "engineering lead", "engineering director"],
    "qa engineer":        ["quality assurance", "test engineer", "sdet", "automation engineer"],
    "security engineer":  ["appsec engineer", "security researcher", "infosec engineer"],
    "blockchain developer":["web3 developer", "smart contract developer", "solidity developer"],
}

# ══════════════════════════════════════════════════════════════════════════════
#  DOMAIN TERMS  (industry / sector vocabulary)
# ══════════════════════════════════════════════════════════════════════════════

DOMAIN_TERMS: dict[str, list[str]] = {
    "fintech":          ["finance", "financial technology", "banking", "payments", "lending",
                         "insurtech", "wealthtech", "regtech", "paytech", "neobank"],
    "healthtech":       ["healthcare", "health technology", "medical", "clinical", "medtech",
                         "digital health", "telehealth", "telemedicine", "biotech", "pharma",
                         "health informatics", "ehr", "emr"],
    "edtech":           ["education", "e-learning", "elearning", "learning technology",
                         "educational technology", "online learning", "lms", "mooc"],
    "ecommerce":        ["retail", "online shopping", "marketplace", "d2c", "dtc",
                         "retail technology", "commerce", "shoptech"],
    "saas":             ["software as a service", "b2b software", "cloud software",
                         "subscription software", "platform as a service", "paas"],
    "gaming":           ["game development", "gamedev", "video games", "mobile gaming",
                         "esports", "game studio", "interactive entertainment"],
    "cybersecurity":    ["security", "infosec", "information security", "appsec",
                         "devsecops", "cyber", "threat intelligence", "soc"],
    "blockchain":       ["web3", "crypto", "cryptocurrency", "defi", "nft", "dapp",
                         "decentralised finance", "decentralized finance", "dao"],
    "adtech":           ["advertising technology", "programmatic", "dsp", "ssp", "ad exchange",
                         "ad server", "digital advertising", "martech"],
    "legaltech":        ["legal technology", "lawtech", "legal ai", "contract tech",
                         "compliance tech", "regtech"],
    "proptech":         ["property technology", "real estate tech", "realestate",
                         "smart building", "facility management"],
    "agritech":         ["agriculture technology", "precision farming", "agri",
                         "smart farming", "food tech"],
    "hrtech":           ["human resources technology", "hr software", "talent tech",
                         "recruitment technology", "workforce management"],
    "traveltech":       ["travel technology", "hospitality tech", "ota", "booking platform"],
    "logistics":        ["supply chain", "supply chain technology", "last mile", "fleet management",
                         "warehouse management", "wms", "freight tech"],
    "automotive":       ["automotive technology", "connected car", "autonomous vehicle",
                         "ev", "electric vehicle", "adas", "fleet"],
    "iot":              ["internet of things", "connected devices", "smart devices",
                         "edge computing", "embedded iot", "industrial iot", "iiot"],
    "aerospace":        ["aerospace engineering", "defence", "defense", "avionics",
                         "satellite", "space tech"],
    "climate":          ["climate tech", "cleantech", "clean energy", "renewable energy",
                         "sustainability tech", "green tech", "net zero"],
}

# ══════════════════════════════════════════════════════════════════════════════
#  ROLE TERMS
# ══════════════════════════════════════════════════════════════════════════════

ALL_ROLE_TERMS: set[str] = {
    # Engineering
    "software engineer", "swe", "software developer", "sde", "developer",
    "frontend developer", "frontend engineer", "ui developer", "ui engineer",
    "backend developer", "backend engineer", "api developer",
    "fullstack developer", "fullstack engineer", "full stack developer",
    "mobile developer", "ios developer", "android developer",
    "ml engineer", "machine learning engineer", "ai engineer", "ai researcher",
    "data scientist", "data analyst", "data engineer", "analytics engineer",
    "devops engineer", "platform engineer", "infrastructure engineer",
    "cloud engineer", "cloud architect", "solutions architect",
    "security engineer", "appsec engineer", "security researcher",
    "embedded engineer", "firmware engineer", "hardware engineer",
    "qa engineer", "sdet", "test engineer", "automation engineer",
    "blockchain developer", "web3 developer", "smart contract developer",
    "game developer", "unity developer", "unreal developer",
    "tech lead", "technical lead", "lead engineer", "staff engineer",
    "principal engineer", "distinguished engineer",
    "engineering manager", "director of engineering",
    "vp of engineering", "cto", "chief technology officer",
    "site reliability engineer", "sre",

    # Data
    "data scientist", "senior data scientist", "lead data scientist",
    "data analyst", "business analyst", "bi analyst",
    "data engineer", "senior data engineer",
    "analytics engineer", "machine learning researcher",
    "quantitative analyst", "quant", "quant researcher",
    "research scientist", "applied scientist",

    # Product & Design
    "product manager", "pm", "senior product manager", "group product manager",
    "product owner", "po", "chief product officer", "cpo",
    "ux designer", "ui designer", "product designer",
    "ux researcher", "user researcher", "design researcher",
    "design lead", "head of design", "vp design",
    "content designer", "ux writer", "content strategist",
    "visual designer", "graphic designer", "motion designer",

    # Management & Leadership
    "engineering manager", "em", "senior engineering manager",
    "director of engineering", "vp engineering",
    "head of engineering", "head of product", "head of data",
    "cto", "cpo", "cio", "chief information officer",
    "ceo", "co-founder", "founder",

    # Specialist
    "blockchain developer", "web3 developer", "solidity developer",
    "ai safety researcher", "mlops engineer", "platform engineer",
    "developer advocate", "developer relations", "devrel",
    "technical program manager", "tpm", "program manager",
    "scrum master", "agile coach", "delivery manager",
    "solutions engineer", "pre-sales engineer", "sales engineer",
    "implementation engineer", "integration engineer",
    "support engineer", "customer success engineer",
}

# ══════════════════════════════════════════════════════════════════════════════
#  EXPERIENCE PATTERNS
# ══════════════════════════════════════════════════════════════════════════════

EXPERIENCE_PATTERNS: dict[str, list[str]] = {
    "entry":     [
        r"\b(0[-–]?[12])\s*years?\b",
        r"\b(fresher|fresh\s+graduate|entry[\s-]level|junior|intern|trainee|graduate)\b",
        r"\b0[-–]2\s*years?\b",
    ],
    "mid":       [
        r"\b([23][-–]?[45])\s*years?\b",
        r"\b(mid[\s-]level|intermediate|associate|3[-–]5\s*years?)\b",
    ],
    "senior":    [
        r"\b([5-9]|10)\s*\+?\s*years?\b",
        r"\b(senior|sr\.?|lead|principal|staff|expert|specialist)\b",
    ],
    "executive": [
        r"\b(1[0-9]|20)\s*\+?\s*years?\b",
        r"\b(director|vp|vice\s+president|head\s+of|c-?level|cto|cpo|cio)\b",
    ],
}

# ══════════════════════════════════════════════════════════════════════════════
#  DERIVED LOOKUPS  (built automatically — do not edit manually)
# ══════════════════════════════════════════════════════════════════════════════

# Reverse lookup: alias → canonical
ALIAS_TO_CANONICAL: dict[str, str] = {}
for _canonical, _aliases in SKILL_SYNONYMS.items():
    for _alias in _aliases:
        ALIAS_TO_CANONICAL[_alias.lower()] = _canonical

# All known skill terms (canonical + all aliases, for entity extraction)
ALL_SKILL_TERMS: set[str] = set()
ALL_SKILL_TERMS.update(s.lower() for s in CORE_SKILLS)
ALL_SKILL_TERMS.update(s.lower() for s in SECONDARY_SKILLS)
ALL_SKILL_TERMS.update(SKILL_SYNONYMS.keys())
for _aliases in SKILL_SYNONYMS.values():
    ALL_SKILL_TERMS.update(a.lower() for a in _aliases)

# All domain surface forms
ALL_DOMAIN_TERMS: set[str] = set()
for _domain, _variants in DOMAIN_TERMS.items():
    ALL_DOMAIN_TERMS.add(_domain)
    ALL_DOMAIN_TERMS.update(_variants)

# Protected terms: never spell-corrected
PROTECTED_TERMS: set[str] = set()
PROTECTED_TERMS.update(ALL_SKILL_TERMS)
PROTECTED_TERMS.update(ALL_DOMAIN_TERMS)
PROTECTED_TERMS.update(r.lower() for r in ALL_ROLE_TERMS)
PROTECTED_TERMS.update([
    "api", "sdk", "ui", "ux", "css", "html", "sql", "nosql", "orm",
    "rest", "graphql", "grpc", "http", "https", "jwt", "oauth",
    "ml", "dl", "ai", "nlp", "cv", "llm", "rag", "swe", "sde",
    "gcp", "aws", "gke", "eks", "ecs", "ec2", "s3", "ci", "cd",
    "cicd", "k8s", "vm", "vpc", "saas", "paas", "iaas",
    "b2b", "b2c", "d2c", "dtc", "defi", "nft", "dao", "dapp",
    "mvp", "okr", "kpi", "sla", "slo", "sli",
])


if __name__ == "__main__":
    print(f"Core skills:      {len(CORE_SKILLS):,}")
    print(f"Secondary skills: {len(SECONDARY_SKILLS):,}")
    print(f"Soft skills:      {len(SOFT_SKILLS):,}")
    print(f"Skill synonyms:   {len(SKILL_SYNONYMS):,} canonical entries")
    print(f"All skill terms:  {len(ALL_SKILL_TERMS):,} (with aliases)")
    print(f"Domain terms:     {len(ALL_DOMAIN_TERMS):,}")
    print(f"Role terms:       {len(ALL_ROLE_TERMS):,}")
    print(f"Protected terms:  {len(PROTECTED_TERMS):,}")
    print(f"Alias mappings:   {len(ALIAS_TO_CANONICAL):,}")

# ══════════════════════════════════════════════════════════════════════════════
#  VOCAB EXPANSION PATCH — appended to hit 2500+ core / 2500+ secondary / 300+ soft
# ══════════════════════════════════════════════════════════════════════════════

_CORE_EXTRA: set[str] = {
    # More languages
    "actionscript","awk","sed","batch","nix","odin","carbon","mojo","gleam","grain",
    "rescript","mint","ballerina","apex (salesforce)","q#","x10","chapel","pony",
    "io","factor","red","rebol","pike","lasso","coldfusion","cfml","livecode",
    "autohotkey","autoit","vbscript","jscript","amos","blitz basic","gambas",
    "monkey x","harbour","freebasic","purebasic","basic","qbasic","gw-basic",
    "turbo pascal","modula-2","oberon","eiffel","simula","algol","pl/1",
    "natural","rpg","mumps","ml","standard ml","caml","agda","coq","idris",
    "lean","isabelle","tla+","alloy","b method","z notation","vdm",

    # More web frontend
    "solid.js","qwik","marko","inferno","mithril.js","surplus","imba",
    "stimulus","hotwire","turbo","unpoly","htmx","hyperscript",
    "web components","custom elements","shadow dom","html imports",
    "css grid","css flexbox","css variables","css animations","css transforms",
    "web animations api","requestanimationframe","intersection observer",
    "mutation observer","resize observer","performance observer",
    "canvas api","svg animation","webgl","webgpu","web audio api",
    "web speech api","web bluetooth","web usb","web nfc","web share",
    "payment request api","credential management","web authn","fido2",
    "indexed db","cache api","background sync","push api","notifications api",
    "geolocation api","device orientation","ambient light sensor",
    "battery api","network information","media session","picture-in-picture",
    "css houdini","css paint api","layout api","animation worklet",
    "css typed om","css properties and values",
    "react server components","server actions","app router","pages router",
    "incremental static regeneration","isr","static site generation","ssg",
    "server-side rendering","ssr","client-side rendering","csr",
    "edge rendering","streaming ssr","suspense","concurrent mode",
    "react 18","react 19","use hook","server components",

    # More backend frameworks
    "litestar","blacksheep","sanic","tornado","aiohttp","starlette",
    "quart","hypercorn","uvicorn","gunicorn","daphne","granian",
    "falcon","bottle","cherrypy","web2py","turbogears","pyramid",
    "zope","plone","wagtail","mezzanine","zinnia",
    "loopback","sails.js","feathers.js","moleculer","strapi","payload cms",
    "keystone.js","prisma (server)","nexus","pothos",
    "actix-web","axum","warp","tide","gotham","nickel","iron",
    "rocket.rs","salvo","poem","viz","axum-extra",
    "ktor","micronaut","quarkus","helidon","vertx","javalin",
    "dropwizard","jersey","resteasy","restlet","cxf",
    "grails","ratpack","sparkjava","jooby","pippo",

    # More databases
    "cockroachdb","yugabyte","vitess","planetscale","neon postgres",
    "supabase postgres","tembo","nile","xata","turso","libsql",
    "singlestore","memsql","voltdb","nuodb","clustrix",
    "mongodb atlas","atlas search","atlas vector search",
    "documentdb","cosmosdb mongodb api",
    "scylladb","keyspaces","astra db",
    "dragonfly","keydb","valkey",
    "typesense","meilisearch","algolia","opensearch",
    "duckdb","motherduck","clickhouse","druid","pinot",
    "timescaledb","questdb","tdengine","kdb+","opentsdb",
    "arangodb","orientdb","dgraph","nebula graph","memgraph",
    "redisgraph","falkordb",
    "etcd","zookeeper","consul (kv)","riak",
    "objectbox","realm","isar","hive (mobile)",

    # More ML/AI tools
    "langsmith","langfuse","helicone","traceloop","agentops",
    "crewai","autogen","semantic router","guardrails ai","nemo guardrails",
    "instructor","outlines","guidance","marvin","mirascope",
    "litellm","openrouter","together ai api","groq api","mistral api",
    "anthropic claude api","openai api","cohere api","ai21 api",
    "replicate api","hugging face inference api","deepinfra","fireworks ai",
    "vllm","text generation inference","tgi","lm studio","ollama",
    "llama.cpp","gpt4all","jan","localai","xinference",
    "axolotl","unsloth","trl","alignment handbook",
    "deepspeed","megatron-lm","fsdp","colossalai",
    "triton (language)","cutlass","flash attention","xformers",
    "bitsandbytes","auto-gptq","awq","exllamav2",
    "whisper","seamless","fastspeech","tacotron","vall-e",
    "musicgen","audiocraft","audioldm","bark",
    "stable diffusion xl","sdxl","lcm","turbo diffusion",
    "controlnet","ip-adapter","lora (diffusion)","dreambooth","textual inversion",
    "animatediff","svd","stable video diffusion","sora","runway","pika",

    # More data tools
    "polars","duckdb","ibis","modin","cudf","rapids",
    "sqlmesh","datafold","recce","piperider",
    "openlineage","marquez","atlas","amundsen","datahub","apache atlas",
    "dbt-unit-testing","dbt-expectations","dbt-audit-helper",
    "hamilton","zenml","bentoml","mlserver","fastapi (ml serving)",
    "evidently","whylogs","fiddler","arthur","arize","aporia",
    "label studio","labelbox","scale data engine","v7 darwin",
    "roboflow","cvat","prodigy","doccano",
    "feast","hopsworks","tecton","vertex feature store",
    "neptune.ai","comet ml","guild.ai","determined ai","clearml",
    "anyscale","modal","runpod","vast.ai","coreweave",

    # More DevOps
    "github actions (advanced)","act","nektos/act",
    "dagger","earthly","bazel","buck2","pants","gradle","maven","sbt",
    "cmake","meson","ninja","make","just","taskfile",
    "skaffold","tilt","garden","telepresence","devspace",
    "okteto","gitpod","codespaces","devcontainer",
    "buildah","kaniko","ko","jib","cloud native buildpacks",
    "cosign","sigstore","syft","grype","trivy","snyk",
    "falco","tetragon","tracee","sysdig",
    "kyverno","gatekeeper","conftest","polaris",
    "network policies","calico","cilium","flannel","weave",
    "cert-manager","external-secrets","sealed-secrets","external-dns",
    "cluster-autoscaler","keda","vpa","goldilocks",
    "velero","kasten","stash","longhorn","rook-ceph",
    "karpenter","node problem detector","node local dns",
    "kubecost","opencost","infracost",
    "pulumi automation api","cdktf","aws cdk python","aws cdk typescript",
    "cloudformation (advanced)","sam (advanced)","serverless framework","sst",
    "cdk8s","helmfile","helmsman","flux2","weave gitops",
    "argo workflows","argo events","argo rollouts","argo cd",
    "tekton pipelines","tekton triggers","tekton catalog",
    "crossplane","kro","kratix",

    # More security
    "secret scanning","code scanning","dependency review",
    "dependabot","renovate","snyk open source","whitesource","mend",
    "codeql","semgrep","sonarqube","sonarcloud","veracode","checkmarx",
    "burp suite professional","zap","nikto","sqlmap","hydra",
    "aircrack-ng","hashcat","john the ripper","mimikatz","bloodhound",
    "responder","impacket","crackmapexec","covenant","cobalt strike",
    "osint","maltego","shodan","censys","greynoise",
    "pwntools","radare2","ghidra","ida pro","binary ninja","cutter",
    "gdb","lldb","peda","pwndbg","gef",
    "frida","objection","jadx","apktool","mobsf",
    "cloud security posture management","cspm","cwpp","cnapp",
    "prisma cloud","wiz","lacework","orca","aqua security",
    "hashicorp vault","cyberark","delinea","beyondtrust",
    "zscaler","netskope","cato networks","cloudflare zero trust",
    "okta","auth0","ping identity","sailpoint","saviynt",
    "splunk soar","palo alto xsoar","ibm qradar soar",
    "crowdstrike falcon","sentinelone","microsoft defender",
    "palo alto networks","fortinet","checkpoint","cisco",

    # More frontend specifics
    "react hooks","useeffect","usestate","usecontext","usereducer","usememo","usecallback","useref",
    "react suspense","react lazy","react portal","react error boundary",
    "react testing library","enzyme","react devtools",
    "vue composition api","vue options api","vuex","pinia",
    "vue router","vue devtools","vueuse","vitest (vue)",
    "angular signals","angular standalone","ngrx","ngxs","akita",
    "angular material","angular cdk","primeng","ngbootstrap",
    "rxjs","observables","subjects","operators","pipeable operators",

    # Additional DevTools
    "cursor ide","windsurf","zed editor","helix editor",
    "warp terminal","fig","atuin","starship",
    "lazygit","delta (diff)","bat","eza","fd","ripgrep","fzf","zoxide",
    "mise","asdf","nvm","pyenv","rbenv","sdkman","volta",
    "direnv","devenv","nix flakes","home-manager",
    "docker compose","docker swarm","podman","lima","colima","orbstack",
    "multipass","devpod","coder",
}

_SECONDARY_EXTRA: set[str] = {
    # More methodologies
    "shape up","dual track agile","continuous discovery","opportunity solution tree",
    "impact mapping","user story mapping","event storming","event modeling",
    "domain storytelling","example mapping","story splitting",
    "six sigma","lean six sigma","value stream mapping","kaizen","5s",
    "theory of constraints","toc","systems thinking","second-order thinking",
    "first principles","inversion","mental models","decision trees",
    "cost-benefit analysis","swot","pestle","porter five forces",
    "wardley mapping","capability mapping","business model canvas",

    # More collaboration tools
    "loom","screen recording","async communication",
    "github discussions","github projects","github milestones",
    "gitlab boards","gitlab milestones","gitlab epics",
    "linear cycles","linear roadmap","linear triage",
    "productboard","aha","roadmunk","prodpad","canny",
    "intercom","zendesk","freshdesk","helpscout","front",
    "calendly","doodle","when2meet","world time buddy",
    "donut","water cooler","gather.town","teamflow",

    # More monitoring
    "sentry","bugsnag","rollbar","airbrake","raygun",
    "elastic apm","instana","appdynamics","dynatrace davis ai",
    "groundcover","coroot","signoz","hyperdx","last9",
    "cronitor","checkly","betteruptime","freshping","statuscake",
    "uptimerobot","pingdom","site24x7",
    "runscope","assertible","datadog synthetic","checkly (api)",
    "chaos engineering","chaos monkey","gremlin","litmus","chaos mesh",
    "game days","disaster recovery drills","blameless postmortems",
    "incident io","firehydrant","blameless","jeli","rootly",

    # More data patterns
    "data mesh","data fabric","data virtualization","data federation",
    "semantic layer","metrics layer","headless bi",
    "data contract","data sla","data observability","data freshness",
    "slowly changing dimension","scd","star schema","snowflake schema","data vault",
    "one big table","obt","wide table","denormalization",
    "lambda architecture","kappa architecture","delta architecture",
    "streaming analytics","real-time analytics","near-real-time","microbatch",
    "data enrichment","data masking","data tokenization","data pseudonymization",
    "master data management","mdm","customer data platform","cdp",
    "reverse etl","operational analytics","data activation",

    # More api patterns
    "api versioning","api deprecation","api lifecycle",
    "api rate limiting","api quota","api throttling","api monetization",
    "api security","api key management","api gateway (pattern)",
    "backend for frontend","bff","api composition","api aggregation",
    "api mocking","api contract testing","consumer-driven contracts",
    "pact testing","spring cloud contract","prism",
    "graphql federation","apollo federation","schema stitching",
    "grpc streaming","bidirectional streaming","grpc gateway",
    "thrift rpc","avro rpc","cap'n proto",
    "hypermedia","hateoas","hal","json-ld","json api","odata",

    # More testing patterns
    "test pyramid","testing trophy","swiss cheese model",
    "mutation testing","pitest","stryker","cosmic ray",
    "fuzz testing","fuzzing","libfuzzer","afl","atheris","jazzer",
    "chaos testing","fault injection","toxiproxy","comcast (tool)",
    "visual regression testing","percy","applitools","chromatic (visual)",
    "snapshot testing","golden master testing",
    "contract testing","pact","spring cloud contract","dredd",
    "api testing","soapui","postman tests","newman","vcrpy","betamax",
    "database testing","testcontainers","docker compose (test)",
    "security testing (appsec)","owasp zap","arachni",
    "accessibility testing","axe","wave","deque axe-core","lighthouse ci",
    "internationalization testing","pseudolocalization",
    "cross-browser testing","browserstack","sauce labs","lambdatest",
    "mobile testing","espresso","xcuitest","appium","detox","maestro",

    # More architecture patterns
    "strangler fig pattern","anticorruption layer","saga pattern",
    "outbox pattern","inbox pattern","transactional outbox",
    "two-phase commit","compensating transactions","distributed saga",
    "bulkhead pattern","retry pattern","timeout pattern","fallback pattern",
    "ambassador pattern","sidecar pattern","adapter pattern","facade pattern",
    "backends for frontends","api gateway pattern","aggregator pattern",
    "choreography","orchestration","process manager","workflow engine",
    "actor model","csp","communicating sequential processes",
    "share nothing architecture","cell-based architecture",
    "modular monolith","majestic monolith","self-contained systems",
    "microfrontend architecture","vertical slice architecture","feature slices",
    "onion architecture","ports and adapters","hexagonal (pattern)",
    "functional core imperative shell","railway oriented programming",

    # More cloud patterns
    "well-architected framework","cloud design patterns",
    "infrastructure as code patterns","policy as code patterns",
    "gitops workflow","pull-based deployment","push-based deployment",
    "progressive delivery","feature toggles","ring deployment","dark launch",
    "shadow mode","traffic mirroring","sticky sessions","session affinity",
    "connection draining","graceful degradation","graceful shutdown",
    "pod disruption budget","pod anti-affinity","topology spread constraints",
    "resource quotas","limit ranges","namespace isolation",
    "multi-tenancy","tenant isolation","namespace-per-tenant","cluster-per-tenant",
    "cost allocation","showback","chargeback","cloud unit economics",
    "spot/preemptible instances","reserved capacity","savings plans",
    "data transfer costs","egress costs","inter-region costs",

    # More soft engineering practices
    "technical debt management","refactoring","code smells","legacy modernisation",
    "strangler fig migration","big bang migration","incremental migration",
    "feature parity","parallel run","shadow testing",
    "code quality metrics","cyclomatic complexity","cognitive complexity",
    "code coverage","branch coverage","mutation score",
    "static analysis","dynamic analysis","runtime verification",
    "formal methods","model checking","abstract interpretation",
    "program synthesis","automated program repair",
    "software supply chain","sbom","sigstore","slsa",
    "open source compliance","license scanning","fossa","blackduck",

    # More frontend patterns
    "micro-frontend orchestration","module federation 2.0",
    "islands architecture","partial hydration","resumability",
    "signals (pattern)","fine-grained reactivity","reactive primitives",
    "derived state","computed state","effects","subscriptions",
    "optimistic updates","pessimistic updates","conflict resolution",
    "offline-first","local-first","crdt","yjs","automerge",
    "collaborative editing","real-time collaboration","presence",
    "cursor sharing","selection sharing","undo/redo stack",

    # More data engineering patterns
    "data lakehouse patterns","medallion architecture","bronze silver gold",
    "data contracts (engineering)","schema evolution","schema registry",
    "confluent schema registry","glue schema registry","apicurio",
    "data serialization","columnar storage","row storage",
    "file format comparison","parquet vs orc vs avro",
    "partition strategy","bucketing","clustering","z-ordering",
    "data skew","data spill","shuffle optimization","broadcast join",
    "incremental processing","full refresh","upsert","merge","slowly changing",
    "exactly-once semantics","at-least-once","at-most-once",
    "backpressure","flow control","watermarks (streaming)","late data handling",

    # More MLOps patterns
    "ml system design","two-phase ml deployment","shadow ml deployment",
    "champion challenger","a/b ml testing","multi-armed bandit",
    "online learning","continual learning pipeline",
    "data pipeline testing","expectation testing","anomaly detection (data)",
    "model cards","model documentation","model governance",
    "feature attribution","shap","lime","integrated gradients","attention viz",
    "calibration","confidence scores","uncertainty quantification",
    "out-of-distribution detection","ood","anomaly detection (ml)",
    "active learning","human-in-the-loop","annotation pipeline",
    "synthetic data generation","data augmentation strategy",
    "cross-validation strategy","stratified sampling","time-series split",

    # More product & business skills
    "product discovery","opportunity sizing","problem framing","jobs to be done",
    "jtbd framework","outcome-based roadmap","now-next-later",
    "hypothesis-driven development","lean startup methodology","build-measure-learn",
    "pirate metrics","aarrr","growth loops","product-led growth","plg",
    "sales-led growth","slg","community-led growth","clg",
    "virality","network effects","flywheel","moat","defensibility",
    "pricing strategy","monetization","freemium","usage-based pricing",
    "value-based pricing","tiered pricing","enterprise pricing",
    "customer acquisition","customer retention","customer expansion",
    "nps","csat","ces","customer effort score","churn rate",
    "monthly active users","mau","daily active users","dau","dau/mau ratio",
    "time to value","ttv","time to wow","activation rate","engagement rate",
}

_SOFT_EXTRA: set[str] = {
    # More communication
    "non-verbal communication","body language","tone management",
    "cross-cultural communication","cultural sensitivity","global mindset",
    "technical storytelling","data storytelling","executive communication",
    "board communication","investor communication","press communication",
    "crisis communication","transparent communication","radical candor",
    "nonviolent communication","nvc","socratic questioning","motivational interviewing",
    "written clarity","editing","proofreading","copy editing","plain language",
    "documentation culture","knowledge management","institutional knowledge",

    # More leadership
    "organisational design","team topology","team structuring",
    "skip-level management","matrix management","dotted-line management",
    "remote team leadership","distributed team leadership","async leadership",
    "psychological safety creation","inclusive leadership","allyship",
    "sponsorship","advocacy","amplification",
    "executive stakeholder management","board management","c-suite communication",
    "budget management","p&l ownership","financial accountability",
    "headcount planning","org design","team scaling","hypergrowth management",
    "firing decisions","pip management","performance improvement",
    "technical strategy","engineering strategy","platform strategy",
    "build vs buy decisions","make vs buy","outsourcing decisions",
    "vendor management","partner management","contract negotiation",
    "hiring bar raising","interview design","structured interviewing",
    "technical interviewing","behavioural interviewing","case interviewing",
    "offer negotiation","compensation design","equity management",

    # More problem solving
    "structured problem solving","mckinsey method","minto pyramid",
    "scqa framework","hypothesis tree","issue tree","logic tree",
    "five whys","fishbone diagram","ishikawa","pareto analysis",
    "decision matrix","pugh matrix","cost-impact matrix",
    "scenario planning","pre-mortem analysis","post-mortem analysis",
    "failure mode analysis","fmea","risk register","risk matrix",
    "trade-off analysis","opportunity cost analysis","npv analysis",
    "break-even analysis","sensitivity analysis","monte carlo (business)",
    "benchmarking","competitive intelligence","market analysis",
    "force field analysis","stakeholder analysis","power-interest grid",

    # More adaptability
    "navigating ambiguity","working with incomplete information",
    "bias for action","speed of iteration","learning agility",
    "unlearning","mindset shift","perspective taking","reframing",
    "cognitive flexibility","set shifting","task switching",
    "dealing with failure","recovering from setbacks","antifragility",
    "stress tolerance","pressure performance","high-stakes delivery",
    "work-life integration","sustainable pace","burnout prevention",
    "self-care","boundary setting","energy management",

    # More collaboration
    "influencing upwards","managing sideways","followership",
    "consensus building","coalition building","getting buy-in",
    "conflict transformation","difficult conversation management",
    "feedback culture","giving feedback","receiving feedback",
    "360 feedback","peer feedback","upward feedback",
    "retrospective facilitation","blameless culture","learning organisation",
    "communities of practice","guild model","chapter model",
    "cross-team dependency management","escalation management",
    "working with legal","working with compliance","working with finance",
    "sales-engineering collaboration","design-engineering collaboration",
    "data-product collaboration","research-engineering collaboration",

    # Cognitive & analytical
    "quantitative reasoning","numerical literacy","statistical intuition",
    "probabilistic thinking","bayesian reasoning","base rate thinking",
    "expected value thinking","optionality thinking","reversibility assessment",
    "complexity management","simplification","abstraction ability",
    "pattern recognition","analogical reasoning","metaphorical thinking",
    "thought experiments","steelman argument","red teaming (thinking)",
    "contrarian thinking","devil's advocate","critical review",
    "synthesis","integration of ideas","cross-domain thinking",
    "generalist thinking","breadth-first exploration","depth-first analysis",

    # Personal effectiveness
    "time boxing","pomodoro","deep work","flow state","maker schedule",
    "calendar management","meeting hygiene","no meeting days",
    "inbox zero","information diet","distraction management",
    "pkm","personal knowledge management","zettelkasten","second brain",
    "writing to think","rubber duck debugging","thinking out loud",
    "accountability partner","peer coaching","mentee skills",
    "self-promotion","visibility","personal branding",
    "career management","career navigation","lateral moves",
    "network building","relationship maintenance","giving back",
}

# Merge expansions into main sets
CORE_SKILLS.update(_CORE_EXTRA)
SECONDARY_SKILLS.update(_SECONDARY_EXTRA)
SOFT_SKILLS.update(_SOFT_EXTRA)


_CORE_EXTRA2: set[str] = {
    # More specific ML/AI frameworks and concepts
    "automl","neural architecture search","nas","hyperparameter optimization","hpo",
    "ensemble methods","bagging","boosting","stacking","blending",
    "federated learning","differential privacy","homomorphic encryption (ml)",
    "causal ml","causal inference","do-calculus","counterfactual","uplift modeling",
    "survival analysis","cox model","kaplan-meier","time-to-event",
    "recommender systems","collaborative filtering","content-based filtering",
    "matrix factorization","als","ncf","two-tower model","retrieval ranking",
    "learning to rank","pointwise","pairwise","listwise","lambdarank",
    "anomaly detection","isolation forest","one-class svm","lof","autoencoder (anomaly)",
    "time series forecasting","prophet","nbeats","nhits","tft","n-hits",
    "graph neural network","gnn","gcn","gat","graphsage","mpnn","graph transformer",
    "knowledge graph embedding","transe","rotate","distmult","complex",
    "multi-task learning","multi-label classification","hierarchical classification",
    "imbalanced classification","smote","class weights","focal loss",
    "metric learning","contrastive learning","triplet loss","arcface","cosface",
    "self-supervised learning","simclr","moco","byol","simsiam","vicreg","mae",
    "vision language model","vlm","clip","blip","flamingo","llava","gpt-4v",
    "embodied ai","robotics ml","sim-to-real","domain randomization",
    "tabular ml","tabnet","ft-transformer","saint","tabtransformer",
    "structured prediction","crf","hmm","seq2seq","pointer network","copy mechanism",
    "monte carlo tree search","mcts","alphazero","muzero","dreamer",
    "model predictive control","mpc","imitation learning","inverse rl",
    "multi-agent reinforcement learning","marl","hierarchical rl","hrl",
    "online rl","offline rl","batch rl","d4rl","conservative q-learning","cql",
    "reward shaping","reward modeling","rlhf","rlaif","dpo","ppo (rl)","sac","td3",
    "a3c","a2c","trpo","natural policy gradient","trust region",
    "gaussian process","gp regression","gp classification","gpytorch","gpflow",
    "variational inference","mean field","expectation maximization","em algorithm",
    "markov chain monte carlo","mcmc","hamiltonian monte carlo","hmc","nuts",
    "normalizing flows","real nvp","glow","nice","iaf","maf","neural spline flow",

    # More data science tools
    "statsmodels","pingouin","lifelines","scikit-survival",
    "imbalanced-learn","category-encoders","feature-engine",
    "optuna","hyperopt","ray tune","nni","auto-sklearn","autokeras","h2o automl",
    "shap","lime","eli5","alibi","interpret ml","what-if tool",
    "pandas-profiling","ydata-profiling","sweetviz","dataprep","dtale",
    "missingno","phik","dython",
    "networkx","igraph","stellargraph","spektral","pyg","dgl",
    "nltk","spacy","gensim","allennlp","flair","stanza","sutime",
    "pytesseract","easyocr","paddleocr","surya","nougat",
    "librosa","pyaudio","soundfile","torchaudio","whisperx",
    "pymupdf","pdfplumber","camelot","tabula-py","pdfminer",
    "beautifulsoup","scrapy","selenium","playwright (python)","httpx","mechanize",
    "arrow","pendulum","dateutil","humanize",
    "pint","sympy","mpmath","gmpy2",
    "pyproj","shapely","fiona","geopandas","rasterio","pyqgis","folium",
    "plotly express","plotly dash","panel","param","holoviews","hvplot",
    "bokeh server","streamlit","gradio","nicegui","fasthtml",

    # More cloud native
    "service catalog (aws)","aws control tower","aws organizations",
    "aws config","aws cloudtrail","aws guardduty","aws inspector","aws macie",
    "aws waf","aws shield","aws network firewall",
    "aws transit gateway","aws private link","aws direct connect",
    "aws datasync","aws transfer family","aws snowball","aws snowcone",
    "aws app runner","aws lightsail","aws beanstalk (advanced)",
    "aws batch","aws glue studio","aws dms","aws schema conversion",
    "gcp binary authorization","gcp artifact analysis","gcp confidential computing",
    "gcp vpc service controls","gcp access context manager",
    "gcp data catalog","gcp dataplex","gcp dataform","gcp looker studio",
    "gcp apigee","gcp api hub","gcp integration connectors",
    "gcp workflows","gcp tasks","gcp scheduler","gcp eventarc",
    "azure api management","azure front door","azure cdn","azure ddos",
    "azure bastion","azure firewall","azure sentinel","azure defender",
    "azure purview","azure information protection",
    "azure logic apps","azure integration services","azure api center",
    "azure managed identity","azure service principal","azure arc",
    "oracle cloud","oci","oracle functions","oracle autonomous database",
    "ibm cloud","ibm watson","ibm cloud pak",
    "alibaba cloud","tencent cloud","huawei cloud",
    "digitalocean","linode","akamai cloud","vultr","hetzner",
    "fly.io","railway","render","vercel","netlify","cloudflare workers","cloudflare pages",
    "fastly","akamai","bunny cdn","cloudinary","imgix",

    # More testing specifics
    "jest (advanced)","vitest","bun test","deno test",
    "pytest (advanced)","pytest-asyncio","pytest-django","pytest-fixtures",
    "rspec","minitest","factory bot","vcr.rb",
    "junit 5","testng (advanced)","spock framework","kotlin test",
    "go test","testify","gomock","ginkgo","gomega",
    "rust test","cargo test","proptest","quickcheck (rust)",
    "load testing","k6 (advanced)","locust (advanced)","jmeter (advanced)",
    "grafana k6","artillery (advanced)","vegeta","hey","wrk","ab",
    "chaos engineering tools","gremlin scenarios","litmus experiments",
    "opentelemetry testing","contract testing tools","wiremock","mockoon",
    "api mocking","msw","nock","fetchmock",

    # More programming paradigms
    "functional programming","fp","pure functions","immutability","referential transparency",
    "monads","functors","applicatives","monoids","semigroups","foldable","traversable",
    "category theory (programming)","type theory","dependent types","refinement types",
    "algebraic data types","adt","pattern matching","exhaustive matching",
    "higher-kinded types","hkt","type classes","traits","protocols (programming)",
    "reactive programming","functional reactive programming","frp",
    "logic programming","constraint programming","declarative programming",
    "concurrent programming","parallel programming","async programming",
    "coroutines","fibers","green threads","cooperative multitasking",
    "lock-free programming","wait-free","cas","compare-and-swap","atomic operations",
    "memory models","happens-before","sequential consistency","linearizability",
    "metaprogramming","macros","lisp macros","rust macros","julia macros",
    "reflection","introspection","code generation","templates (c++)","generics",
    "aspect-oriented programming","aop","cross-cutting concerns",
    "protocol-oriented programming","interface segregation","duck typing",

    # More specific tools
    "deno","bun runtime","node18","node20","node22",
    "python 3.11","python 3.12","python 3.13","pypy","graalpy","micropython",
    "jdk 17","jdk 21","graalvm","openjdk","adoptium","amazon corretto",
    "dotnet 8","dotnet 9","dotnet aspire","minimal api","blazor (advanced)",
    "rust 2021","rust 2024","cargo","crates.io","tokio","async-std",
    "go 1.21","go 1.22","go generics","go workspace","go modules",
    "swift 5.9","swift 5.10","swift concurrency","swift package manager",
    "kotlin 2.0","kotlin multiplatform","kotlin native","kmm",
    "scala 3","cats","cats-effect","zio","akka typed","pekko",

    # Databases advanced
    "postgresql 16","postgresql extensions","pg_vector","timescaledb","citus",
    "mysql 8","mysql group replication","mysql innodb cluster",
    "mongodb 7","mongodb atlas search","atlas vector search","mongodb charts",
    "redis 7","redis stack","redisearch","rejson","redisbloom","redistimeseries",
    "cassandra 5","scylladb 6","datastax astra","cassandra lwt",
    "clickhouse cloud","clickhouse replicated mergetree","clickhouse materialized views",
    "apache druid ingestion","druid sql","druid native queries",
    "duckdb extensions","duckdb httpfs","duckdb delta","duckdb iceberg",
    "sqlite (advanced)","sqlite fts5","sqlite rtree","sqlite json",
    "graph databases (advanced)","cypher","gremlin","sparql","dgraph dql",
    "neo4j bloom","neo4j aura","neo4j graph data science",
    "vector search (advanced)","ann algorithms","hnsw","ivfpq","lsh","faiss",
    "pgvector","nmslib","hnswlib","annoy","scann",
}

_SECONDARY_EXTRA2: set[str] = {
    # Security operations
    "threat hunting","threat intelligence","threat modeling (stride)","pasta",
    "diamond model","kill chain","mitre att&ck navigator",
    "ioc","indicator of compromise","ioa","indicator of attack",
    "yara","sigma","snort rules","suricata rules",
    "memory forensics","disk forensics","network forensics","mobile forensics",
    "volatility framework","autopsy","sleuth kit","ftk","encase",
    "dfir","digital forensics and incident response",
    "soar playbooks","runbook automation","alert triage","escalation procedures",
    "vulnerability disclosure","responsible disclosure","bug bounty",
    "cve management","nvd","exploit database","vulndb",
    "penetration testing report writing","pentest methodology","owasp testing guide",
    "web application security","api security testing","mobile security testing",
    "network penetration testing","internal pentest","external pentest",
    "red team operations","purple team exercises","tabletop exercises",
    "security architecture review","threat modeling workshop",
    "secure sdlc","shift left security","devsecops pipeline","security gates",

    # More data governance
    "data stewardship","data ownership","data democratization",
    "data literacy programs","data culture","self-service data",
    "data mesh principles","domain ownership","data product thinking",
    "data sla management","data freshness sla","data quality sla",
    "column-level security","row-level security","attribute-based access",
    "data classification","sensitivity labels","data tagging",
    "pii detection","pii masking","data de-identification","k-anonymity",
    "l-diversity","t-closeness","differential privacy (data)",
    "consent management platform","preference management",
    "right to erasure implementation","data subject access request","dsar",
    "cross-border data transfer","standard contractual clauses","scc",
    "data residency","data sovereignty","data localization",

    # More platform engineering
    "golden path implementation","paved road tooling","developer self-service",
    "infrastructure self-service","environment as a service","eaas",
    "platform product management","platform roadmap","platform metrics",
    "developer experience metrics","dora metrics","space framework",
    "deployment frequency","lead time for changes","mttr","change failure rate",
    "platform adoption","platform onboarding","platform documentation",
    "internal tooling","developer tools","build systems","release tooling",
    "environment management","ephemeral environments","preview environments",
    "feature environment","staging environment","production-like environments",
    "database provisioning","database as a service","dbaas","managed databases",
    "secret rotation","certificate rotation","key rotation",
    "compliance automation","policy enforcement","guardrails",
    "cost visibility","resource tagging strategy","tag enforcement",
    "quota management","budget alerts","spending anomaly detection",

    # More MLOps specifics
    "model versioning","model artifact management","model packaging",
    "model a/b testing","canary model deployment","shadow model deployment",
    "real-time inference","batch inference","asynchronous inference","streaming inference",
    "model warm-up","model caching","model reuse","model composition",
    "feature freshness","feature reuse","point-in-time correct features",
    "training-serving skew","distribution shift detection","drift monitoring",
    "model retraining triggers","automated retraining","continuous training","ct",
    "ml pipeline testing","unit tests (ml)","integration tests (ml)","data tests (ml)",
    "ml experiment management","experiment reproducibility","model provenance",
    "ml governance","model risk management","model validation","model audit",
    "responsible ai","fairness","bias detection","bias mitigation",
    "model explainability","global explanations","local explanations",
    "counterfactual explanations","contrastive explanations",
    "human-in-the-loop ml","active learning pipeline","label efficiency",
    "data-centric ai","dataset curation","data flywheel",

    # More frontend engineering
    "performance budget","bundle analysis","tree shaking","code splitting",
    "lazy loading","prefetching","preloading","dns prefetch","preconnect",
    "critical rendering path","above the fold","layout shift","cls",
    "largest contentful paint","lcp","first input delay","fid","inp",
    "time to first byte","ttfb","first contentful paint","fcp",
    "total blocking time","tbt","speed index","performance score",
    "image optimization","next-gen formats","webp","avif","jpeg xl",
    "responsive images","srcset","picture element","art direction",
    "font optimization","font subsetting","font loading strategies","fout","foit","foft",
    "caching strategies","cache-control","etag","last-modified","stale-while-revalidate",
    "service worker caching","cache-first","network-first","stale-while-revalidate (sw)",
    "offline support","background sync","periodic background sync",
    "web vitals monitoring","rum","real user monitoring","synthetic monitoring",
    "a/b testing (frontend)","feature flags (frontend)","experimentation platform",

    # More backend engineering
    "database connection management","connection pool tuning","query optimization",
    "n+1 problem","eager loading","lazy loading (db)","batch loading","dataloader",
    "database index types","btree index","hash index","gin index","gist index","brin",
    "full-text search implementation","tsvector","tsquery","pg_search",
    "geospatial queries","postgis operations","spatial indexing","r-tree",
    "json operations in db","jsonb","json path","json aggregation",
    "stored procedures performance","function inlining","planner hints",
    "partitioning strategies","range partition","hash partition","list partition",
    "table inheritance","schema-per-tenant","row-level tenant","shared schema",
    "replication lag","read replica routing","cdc (backend)","logical replication",
    "database vacuum","analyze","autovacuum tuning","bloat management",
    "advisory locks","row-level locks","table locks","deadlock prevention",
    "optimistic locking","pessimistic locking","mvcc","serializable isolation",

    # More distributed systems
    "consensus algorithms","raft","paxos","viewstamped replication","zab",
    "vector clocks","lamport timestamps","logical clocks","hybrid logical clocks",
    "gossip protocol","phi accrual failure detector","swim protocol",
    "consistent hashing","virtual nodes","vnodes","chord","kademlia",
    "distributed hash table","dht","peer-to-peer","p2p",
    "leader election","distributed locks","redlock","zookeeper locks",
    "distributed transactions","xa protocol","2pc","3pc",
    "idempotency","idempotency keys","at-least-once delivery","deduplication",
    "message ordering","total order broadcast","causal broadcast","fifo broadcast",
    "backpressure handling","flow control (distributed)","rate limiting (distributed)",
    "circuit breaker implementation","hystrix","resilience4j","polly",
    "service discovery","client-side discovery","server-side discovery",
    "dns-based discovery","consul service discovery","eureka","nacos",
    "configuration management","config server","apollo config","nacos config",
    "distributed caching","cache invalidation","cache stampede","dog-pile effect",
    "cache warming","read-through","write-through","write-behind","write-around",

    # More api design
    "api first development","design first","code first","contract first",
    "api design guidelines","naming conventions","versioning strategy","deprecation policy",
    "api pagination","cursor pagination","offset pagination","keyset pagination",
    "api filtering","odata filtering","graphql filtering","rsql",
    "api sorting","multi-sort","api field selection","sparse fieldsets",
    "api documentation standards","openapi 3.1","asyncapi","graphql sdl",
    "api testing strategy","api contract testing","api regression testing",
    "api performance testing","api security testing","api chaos testing",
    "api monitoring","api analytics","api usage tracking","api developer portal",
    "api sdk generation","openapi generator","swagger codegen","kiota",
    "api backward compatibility","api breaking changes","api evolution",
    "hypermedia apis","hateoas","level 3 rest","richardson maturity model",
    "event-driven api","webhooks design","event schema design","event catalog",

    # More engineering culture
    "engineering excellence","technical craft","software craftsmanship",
    "clean code practices","readable code","self-documenting code",
    "code review best practices","review checklist","review guidelines",
    "on-call practices","on-call rotation","escalation policy","alert fatigue",
    "postmortem culture","blameless postmortem","learning reviews","incident reviews",
    "tech debt triage","tech debt tracking","tech debt board","paydown planning",
    "architecture review board","arb","design reviews","rfc process",
    "engineering metrics","velocity tracking","cycle time","lead time",
    "team health check","team radar","retrospective formats","futurespective",
    "engineering branding","employer brand","developer brand","thought leadership",
    "open source strategy","inner source strategy","contribution guidelines",
    "dependency management","version pinning","lockfiles","reproducible builds",
    "release management","semantic versioning","changelog","release notes",
    "feature lifecycle","feature deprecation","sunset policy","migration guides",

    # More specific technologies
    "webassembly (advanced)","wasm bindgen","wasm-pack","wasmer","wasmtime","wasi",
    "grpc-web","connect protocol","buf","protoc","grpc gateway (advanced)",
    "graphql subscriptions","graphql mutations","graphql directives","graphql scalars",
    "relay","graphql code generator","codegen","type-graphql","nexus (graphql)",
    "tinybird","motherduck (advanced)","hydra data","parasail","xata (advanced)",
    "neon (advanced)","planetscale vitess","tidb cloud","cockroachdb serverless",
    "alloy db","spanner (advanced)","bigtable (advanced)","firestore (advanced)",
    "dynamodb streams","dynamodb global tables","dynamodb on-demand",
    "aurora serverless v2","aurora global","rds proxy","rds read replica",
    "redshift ra3","redshift spectrum","redshift serverless","redshift ml",
    "bigquery omni","bigquery bi engine","bigquery connected sheets","bigquery ml",
    "snowflake marketplace","snowflake data sharing","snowflake cortex","snowpark",
    "databricks unity catalog","databricks delta live tables","databricks workflows",
    "dbt semantic layer","dbt mesh","dbt cloud ide","dbt explorer",
}

CORE_SKILLS.update(_CORE_EXTRA2)
SECONDARY_SKILLS.update(_SECONDARY_EXTRA2)

# ── Geography Protection ─────────────────────────────────────────────────────
# Prevent SymSpell from corrupting major tech hubs into general English words.
PROTECTED_GEOGRAPHY = {
    "noida", "bangalore", "bengaluru", "pune", "gurgaon", "gurugram", "hyderabad",
    "chennai", "mumbai", "delhi", "haryana", "karnataka", "maharashtra", "india"
}

# Rebuild derived lookups after all merges
ALL_SKILL_TERMS = set()
ALL_SKILL_TERMS.update(s.lower() for s in CORE_SKILLS)
ALL_SKILL_TERMS.update(s.lower() for s in SECONDARY_SKILLS)
ALL_SKILL_TERMS.update(SKILL_SYNONYMS.keys())
for _aliases in SKILL_SYNONYMS.values():
    ALL_SKILL_TERMS.update(a.lower() for a in _aliases)

PROTECTED_TERMS = set()
PROTECTED_TERMS.update(ALL_SKILL_TERMS)
PROTECTED_TERMS.update(s.lower() for s in [d for dlist in DOMAIN_TERMS.values() for d in dlist])
PROTECTED_TERMS.update(DOMAIN_TERMS.keys())
PROTECTED_TERMS.update(r.lower() for r in ALL_ROLE_TERMS)
PROTECTED_TERMS.update(PROTECTED_GEOGRAPHY)
PROTECTED_TERMS.update(["api","sdk","ui","ux","css","html","sql","nosql","orm",
    "rest","graphql","grpc","http","https","jwt","oauth","ml","dl","ai","nlp","cv",
    "llm","rag","swe","sde","gcp","aws","gke","eks","ecs","ec2","s3","ci","cd",
    "cicd","k8s","vm","vpc","saas","paas","iaas","b2b","b2c","d2c","dtc",
    "defi","nft","dao","dapp","mvp","okr","kpi","sla","slo","sli"])

print(f"FINAL Core skills:      {len(CORE_SKILLS):,}")
print(f"FINAL Secondary skills: {len(SECONDARY_SKILLS):,}")
print(f"FINAL Soft skills:      {len(SOFT_SKILLS):,}")
print(f"FINAL All skill terms:  {len(ALL_SKILL_TERMS):,}")
print(f"FINAL Protected terms:  {len(PROTECTED_TERMS):,}")
