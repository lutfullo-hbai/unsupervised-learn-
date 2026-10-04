"""M0.T3 — Namuna PDF'larni yaratish.

Skript namunaviy PDF'lar yaratadi (PyMuPDF bilan, tashqi fayl yuklamasdan).
Har bir PDF aniq mavzularga ega bo'lib, clustering sifatini tekshirish uchun
mo'ljalgan.

Ishga tushirish:  .venv/bin/python scripts/make_sample_pdfs.py
"""

from pathlib import Path

import pymupdf

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

FONT_CANDIDATES = (
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
    Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),
)


def _pick_fontfile() -> str | None:
    """Kirill va o'zbekcha harflarni qo'llaydigan shriftni topadi.

    PyMuPDF'ning `helv` (Helvetica) base14 shrifti **faqat lotin** harflarni
    qo'llaydi — undagi ruscha matn `??????` bo'lib chiqadi va tarjima
    moduli uni tarjima qila olmaydi. Shu sabab shrift tashqaridan
    beriladi. Topilmasa `None` qaytariladi (base14, faqat lotin).
    """
    for candidate in FONT_CANDIDATES:
        if candidate.is_file():
            return str(candidate)
    return None


FONTFILE = _pick_fontfile()


def _page_text(page_no: int, total: int, heading: str, body: str, footer: str) -> str:
    return (
        f"Chapter {page_no}. {heading}\n"
        f"{'=' * (len(heading) + 12)}\n\n"
        f"{body}\n\n"
        f"{footer}\n"
        f"Page {page_no} of {total}\n"
    )


def build_pdf(path: Path, title: str, pages: list[tuple[str, str]]) -> None:
    """pages: [(heading, body), ...]"""
    doc = pymupdf.open()
    total = len(pages)
    footer = f"{title} - Technical Report"

    for i, (heading, body) in enumerate(pages, start=1):
        page = doc.new_page()
        text = _page_text(i, total, heading, body, footer)
        if FONTFILE is None:
            page.insert_textbox(
                pymupdf.Rect(60, 60, 540, 780),
                text,
                fontsize=11,
                fontname="helv",
                align=0,
            )
        else:
            page.insert_textbox(
                pymupdf.Rect(60, 60, 540, 780),
                text,
                fontsize=11,
                fontname="F0",
                fontfile=FONTFILE,
                align=0,
            )
    doc.save(str(path))
    doc.close()


# ----------------------------------------------------------------------
# 1) QISQA PDF (3 bet) — machine learning asoslari
# ----------------------------------------------------------------------
SHORT_PAGES = [
    (
        "Introduction to Supervised Learning",
        "Supervised learning trains a model on labelled examples. The dataset contains "
        "input features and a target label for every row. The model learns a mapping "
        "from features to labels using a loss function and an optimisation algorithm. "
        "Common algorithms include linear regression, logistic regression, decision "
        "trees, support vector machines and neural networks. Training uses a loss "
        "function such as mean squared error or cross entropy. Gradient descent "
        "updates the model parameters by iterating over the training data.",
    ),
    (
        "Unsupervised Learning and Clustering",
        "Unsupervised learning works with unlabelled data. The goal is to discover "
        "hidden structure inside the dataset without any ground truth labels. "
        "Clustering is the most common unsupervised task. K-Means partitions data "
        "into k groups by minimising the within-cluster sum of squared distances. "
        "The algorithm alternates between assignment of points to centroids and "
        "recomputation of centroid positions. Other methods include DBSCAN, "
        "hierarchical clustering and Gaussian mixture models. Evaluation relies on "
        "silhouette score, inertia and manual inspection because no ground truth "
        "exists for unsupervised tasks.",
    ),
    (
        "Evaluation of Unsupervised Models",
        "Silhouette score compares the mean intra-cluster distance with the mean "
        "distance to the nearest cluster. Values range from minus one to one. Higher "
        "values indicate better separated clusters. Inertia measures the sum of "
        "squared distances of points to their assigned centroid and always decreases "
        "when the number of clusters grows. The elbow method inspects the inertia "
        "curve to choose k. Stability is verified by running the algorithm with "
        "several random seeds and comparing the resulting partitions.",
    ),
]

# ----------------------------------------------------------------------
# 2) ORTA PDF (8 bet) — distributed systems (texnik hujjat uslubida)
# ----------------------------------------------------------------------
MEDIUM_PAGES = [
    (
        "Cluster Computing Fundamentals",
        "A cluster is a set of independent computers that work together as a single "
        "system. Cluster computing improves throughput and availability compared with "
        "a single machine. Nodes are connected by a high speed network. Job "
        "schedulers distribute tasks across available nodes. Failover mechanisms "
        "detect node failures and reschedule the affected tasks. Storage systems "
        "such as distributed file systems provide shared access to data across the "
        "entire cluster. Capacity planning estimates the required number of nodes "
        "based on workload characteristics and service level objectives.",
    ),
    (
        "Data Replication Strategies",
        "Replication stores multiple copies of data on different nodes. Primary-backup "
        "replication keeps one authoritative copy and one or more backups. Multi-master "
        "replication allows writes on several nodes and requires conflict resolution. "
        "Synchronous replication guarantees consistency at the cost of latency. "
        "Asynchronous replication improves write performance but may lose the latest "
        "updates during a failure. Quorum based replication combines both approaches "
        "and tolerates a certain number of node failures.",
    ),
    (
        "Load Balancing Strategies",
        "Load balancing distributes incoming requests across servers. Round robin "
        "sends requests in a fixed order. Weighted round robin assigns different "
        "capacities to servers. Least connections selects the server with the fewest "
        "active connections. Consistent hashing maps keys to servers so that a small "
        "number of nodes change when the cluster scales. Health checks remove failed "
        "nodes from the rotation. Rate limiting protects the cluster from overload.",
    ),
    (
        "Consensus Protocols",
        "Consensus protocols allow a set of nodes to agree on a value despite "
        "failures. Paxos is the classical protocol and guarantees safety under "
        "partition. Raft was designed for understandability and splits the problem "
        "into leader election, log replication and safety. Quorum systems require a "
        "majority of nodes to agree. Byzantine fault tolerant protocols tolerate "
        "arbitrary faulty nodes but require more participants. Consensus is used by "
        "coordination services such as ZooKeeper and etcd.",
    ),
    (
        "Container Orchestration",
        "Container orchestration systems schedule containers across a cluster of "
        "machines. Kubernetes represents desired application state as declarative "
        "objects. Pods group containers that share a network namespace and storage. "
        "Replica sets maintain the desired number of pods for a deployment. Services "
        "provide stable network endpoints and load balancing. Config maps and secrets "
        "store application configuration. Horizontal pod autoscaling adjusts the pod "
        "count according to observed resource usage.",
    ),
    (
        "Observability and Monitoring",
        "Observability describes how well a system can be understood from its "
        "external outputs. Metrics are numeric time series such as latency, error "
        "rate and throughput. Logs record discrete events with timestamps and "
        "attributes. Traces follow a request across multiple services. Distributed "
        "tracing uses identifiers to correlate data from different services. "
        "Alerting rules notify operators when service level indicators degrade. "
        "Dashboards visualise trends over time and help capacity planning.",
    ),
    (
        "Storage Engines",
        "Log structured merge trees are write optimised and support sequential scans. "
        "They append entries to an immutable log and merge them in the background. "
        "B-trees support efficient random reads and writes and suit point queries. "
        "Write ahead logging guarantees durability before data reaches the main store. "
        "Compaction strategies such as level and tiered reduce read amplification at "
        "the cost of write amplification. Columnar storage organises data by column "
        "and improves analytical query performance.",
    ),
    (
        "Security in Distributed Systems",
        "Transport layer security encrypts network communication between services. "
        "Mutual authentication verifies both peers using certificates. Role based "
        "access control limits actions according to identities. Secrets management "
        "stores credentials outside the source code repository. Audit logs record "
        "sensitive operations for later inspection. Principle of least privilege "
        "grants only the minimum permissions required for each component.",
    ),
]

# ----------------------------------------------------------------------
# 3) KO'P MAVZULI PDF (12 bet) — aralash: ML, DB, security, cloud
# ----------------------------------------------------------------------
MULTI_TOPIC_PAGES = [
    (
        "Neural Network Fundamentals",
        "A neural network is a composition of layers of artificial neurons. Each "
        "connection carries a weight that is adjusted during training. Forward "
        "propagation computes the output of each layer. Backpropagation applies the "
        "chain rule to compute gradients. Activation functions such as ReLU and "
        "softmax introduce non linearity. Gradient descent optimises the weights. "
        "Regularisation techniques such as dropout and weight decay reduce overfitting.",
    ),
    (
        "Convolutional Architectures",
        "A convolutional layer applies small filters across the input. Filters slide "
        "over the image and detect local features such as edges and textures. "
        "Pooling layers reduce spatial resolution. Residual connections help train "
        "very deep networks. Architectures such as ResNet and EfficientNet dominate "
        "image classification benchmarks. Data augmentation such as cropping, flipping "
        "and colour jitter improves generalisation. Transfer learning reuses "
        "pretrained weights for smaller datasets.",
    ),
    (
        "Relational Database Design",
        "A relational database stores data in tables with rows and columns. Primary "
        "keys identify rows uniquely. Foreign keys enforce referential integrity "
        "between tables. Normalisation reduces redundancy and anomalies. The third "
        "normal form eliminates transitive dependency. Indexes accelerate reads but "
        "slow down writes. Query optimisers choose an execution plan based on cost "
        "estimation.",
    ),
    (
        "Query Optimisation",
        "A query optimiser evaluates multiple execution plans and selects the "
        "cheapest one. Cost estimation relies on statistics about table sizes, "
        "index selectivity and value distribution. Join algorithms include nested "
        "loop, hash join and merge join. Sequential scan reads the whole table while "
        "index scan follows an index structure. Execution plans form a tree of "
        "operators. Explain statements reveal the chosen plan and estimated costs.",
    ),
    (
        "Buffer Pool Management",
        "The buffer pool keeps frequently accessed pages in memory. PostgreSQL uses "
        "a shared buffer pool with clock based replacement. Dirty pages are written "
        "to disk before eviction. Checkpoints flush dirty buffers to disk. Memory "
        "allocation follows a slab or arena strategy. Operating system page cache "
        "can double as an additional buffer layer when configured appropriately.",
    ),
    (
        "Authentication Mechanisms",
        "Authentication verifies the identity of a user. Password based "
        "authentication should store adaptive hashed passwords with a per user salt. "
        "Rate limiting slows down brute force attacks. Multi factor authentication "
        "combines several independent proofs of identity. Token based authentication "
        "issues short lived signed tokens after a successful login. Refresh tokens "
        "allow renewal without re-entering credentials.",
    ),
    (
        "Authorisation Models",
        "Authorisation decides what an authenticated identity may perform. Role based "
        "access control groups permissions into roles. Attribute based access control "
        "evaluates subject, resource and context attributes. Principle of least "
        "privilege grants only required permissions. Capability based models restrict "
        "access to specific resources. Access control lists store permissions per "
        "resource. Audit trails record every permission change.",
    ),
    (
        "Serverless Architecture",
        "Serverless computing hides infrastructure management from the developer. "
        "Functions execute short lived code in response to events. Cold starts add "
        "latency when a new instance is initialised. Concurrency limits protect the "
        "platform from overload. Billed resources are measured in executions and "
        "duration rather than provisioned capacity. Containers offer a middle ground "
        "between virtual machines and functions.",
    ),
    (
        "Container Fundamentals",
        "A container packages an application with its dependencies into a single "
        "artefact. Containers share the host kernel which makes them lighter than "
        "virtual machines. Layers form an immutable filesystem stack. Namespaces "
        "isolate processes, networking and mounts. Control groups limit CPU, memory "
        "and disk usage. Image building follows best practices such as multi stage "
        "builds and minimal base images.",
    ),
    (
        "Scaling Strategies",
        "Vertical scaling increases capacity of a single machine and is limited by "
        "hardware. Horizontal scaling adds machines and improves fault tolerance. "
        "Load based scaling adjusts capacity according to traffic. Predictive scaling "
        "uses historical patterns to provision resources in advance. Autoscaling "
        "policies define scaling out and scaling in thresholds with cooldown "
        "periods. Capacity must account for peak rather than average load.",
    ),
    (
        "Message Queue Patterns",
        "A message queue decouples producers from consumers. The publish subscribe "
        "pattern broadcasts messages to multiple consumers. Work queues distribute "
        "messages among competing consumers. Dead letter queues store messages that "
        "fail repeatedly. Message brokers such as Kafka provide partitioned logs "
        "with at least once delivery semantics. Exactly once delivery requires "
        "transactional producers and idempotent consumers.",
    ),
    (
        "Monitoring Costs",
        "Cost monitoring tracks expenditure per service and per team. Resource "
        "tagging enables allocation of charges. Rightsizing removes oversized "
        "instances that consume budget without extra benefit. Reserved instances "
        "reduce cost for steady workloads while spot instances suit fault tolerant "
        "batch jobs. Budget alerts notify operators before unexpected spending. "
        "Unit economics measure cost per request or per user.",
    ),
]

# ----------------------------------------------------------------------
# 4) ARALASH TILLI PDF (6 bet) — o'zbek + ingliz + rus (M-T uchun)
# ----------------------------------------------------------------------
MIXED_LANGUAGE_PAGES = [
    (
        "Sun'iy intellekt va machine learning",
        "Sun'iy intellekt (AI) — bu mashinalarning inson aqliga yaqin xatti-harakat "
        "namoyish etishi. Machine learning esa algoritmlarni misol (data) yordamida "
        "o'qitish jarayoni. Model parametrlari training jarayonida avtomatik "
        "yangilanadi. Chuqur o'rganish (deep learning) ko'p qatlamli neyron tarmoqlar "
        "asosida quriladi. Ushbu yondashuv tasvirlarni tasniflash, tabiiy tilni "
        "qayta ishlash va tavsiya tizimlarida keng qo'llaniladi.",
    ),
    (
        "Ma'lumotlarni tahlil qilish (Data Science)",
        "Data science — statistika, dasturlash va soha bilimini birlashtirgan fan. "
        "Tahlil bosqichlari: ma'lumot yig'ish, tozalash, vizualizatsiya va modelling. "
        "Exploratory data analysis (EDA) orqali ma'lumotning tuzilishi o'rganiladi. "
        "Python ilmiy hisob-kitoblar uchun eng keng tarqalgan til hisoblanadi. "
        "pandas kutubxonasi jadval ko'rinishidagi ma'lumotlarni boshqaradi.",
    ),
    (
        "Maшинное обучение (Russian)",
        "Машинное обучение — это раздел искусственного интеллекта, который учит "
        "алгоритмы на данных без явной разметки. Обучение с учителем использует "
        "помеченные примеры. Кластеризация относится к обучению без учителя и "
        "группирует похожие объекты. Метод k-средних является одним из самых "
        "популярных алгоритмов кластеризации.",
    ),
    (
        "Нейронные сети и deep learning",
        "Нейронная сеть состоит из слоёв искусственных нейронов. Прямой проход "
        "вычисляет выход каждого слоя. Обратное распространение вычисляет "
        "градиенты. Функции активации вносят нелинейность. Регуляризация помогает "
        "избежать переобучения. Модель обучается на размеченных данных и затем "
        "применяется к новым примерам.",
    ),
    (
        "Tashqi arxitektura va bulutli hisoblash",
        "Cloud computing — bu ma'lumotlarni va hisoblash resurslarini masofadan "
        "taqdim etish. Asosiy xizmat modellari: IaaS, PaaS va SaaS. Virtual "
        "machines (VM) — bu bitta server emulyatsiyasi. Containers yanada yengilroq "
        "variant hisoblanadi, chunki ular operatsion tizim yadrosini umumiy "
        "ishlatadi. Orchestration tizimlari konteynerlarni avtomatik boshqaradi.",
    ),
    (
        "Klassifikation va regression",
        "Classification — bu turlarni belgilash masalasi, masalan spam yoki "
        "spam emas. Regression — bu uzluksiz qiymatni bashorat qilish, masalan "
        "narxni yoki haroratni. Decision trees, random forest va gradient boosting "
        "klassik usullar hisoblanadi. Model sifatini baholash uchun train/test "
        "split va kross validatsiya ishlatiladi. Overfitting — model training "
        "ma'lumotini yod qilib, yangi datada yomon ishlash holati.",
    ),
]


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)

    targets = [
        ("01_short_ml_basics.pdf", "Machine Learning Basics", SHORT_PAGES),
        ("02_medium_distributed_systems.pdf", "Distributed Systems", MEDIUM_PAGES),
        ("03_multi_topic_report.pdf", "Multi Topic Report", MULTI_TOPIC_PAGES),
        ("04_mixed_language.pdf", "Mixed Language Sample", MIXED_LANGUAGE_PAGES),
    ]

    for filename, title, pages in targets:
        path = RAW / filename
        build_pdf(path, title, pages)
        size_kb = path.stat().st_size / 1024
        print(f"  {filename:38s} {len(pages):2d} bet  {size_kb:7.1f} KB")

    print(f"\nTayyor: {RAW}")


if __name__ == "__main__":
    main()