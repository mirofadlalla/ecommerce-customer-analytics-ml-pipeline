---

## 🐳 **13. Deployment & Docker Setup**

### 🚀 **Run the Project inside Docker**

Run the entire FastAPI + ML API service in a Docker container with just a few commands:

---

### 🧱 **1️⃣ Build the Docker image**
```bash
docker build -t ecommerce-fastapi .

### ⚙️ 2️⃣ Run the container
```bash
docker run -d -p 8000:8000 ecommerce-fastapi

### 🧪 3️⃣ Test the API

- Open your browser or API client at:

- 👉 http://localhost:8000/docs

### Or use Postman with the following header:

- X-API-Key: 12345678
