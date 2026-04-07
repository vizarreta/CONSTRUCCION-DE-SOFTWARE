from flask import Flask, request, jsonify

# Create Flask app
app = Flask(__name__)

tasks = []
users = []

# Contadores para generar IDs únicos de forma segura
task_id_counter = 1
user_id_counter = 1

@app.route("/")
def home():
    return "Hello, Flask! Welcome to Session 1."

# ==========================================
#               TASKS API
# ==========================================

@app.route("/tasks", methods=["GET"])
def get_tasks():
    return jsonify({"tasks": tasks})

# NUEVO: Obtener una sola tarea por ID
@app.route("/tasks/<int:task_id>", methods=["GET"])
def get_single_task(task_id):
    task = next((t for t in tasks if t["id"] == task_id), None)
    if task is None:
        return jsonify({"error": "Task not found"}), 404
    return jsonify({"task": task})

# POST - add a new task (Modificado con validación)
@app.route("/tasks", methods=["POST"])
def add_task():
    global task_id_counter
    data = request.json
    content = data.get("content", "").strip()
    
    # VALIDACIÓN: No se pueden crear tareas vacías
    if not content:
        return jsonify({"error": "Task content cannot be empty"}), 400
        
    task = {
        "id": task_id_counter, 
        "content": content, 
        "done": data.get("done", False) # Permite recibir el estado o lo pone en False
    }
    tasks.append(task)
    task_id_counter += 1
    return jsonify({"message": "Task added!", "task": task}), 201

# PUT - update a task by ID (Modificado para actualizar "done")
@app.route("/tasks/<int:task_id>", methods=["PUT"])
def update_task(task_id):
    task = next((t for t in tasks if t["id"] == task_id), None)
    if task is None:
        return jsonify({"error": "Task not found"}), 404
        
    data = request.json
    
    if "content" in data:
        content = data["content"].strip()
        if not content:
            return jsonify({"error": "Task content cannot be empty"}), 400
        task["content"] = content
        
    # NUEVO: Marcar tarea como completada (done=True/False)
    if "done" in data:
        task["done"] = bool(data["done"])
        
    return jsonify({"message": "Task updated!", "task": task})

# DELETE - delete a task by ID (Corregido para evitar errores de índice)
@app.route("/tasks/<int:task_id>", methods=["DELETE"])
def delete_task(task_id):
    global tasks
    task = next((t for t in tasks if t["id"] == task_id), None)
    if task is None:
        return jsonify({"error": "Task not found"}), 404
        
    tasks = [t for t in tasks if t["id"] != task_id]
    return jsonify({"message": "Task deleted!", "task": task})


# ==========================================
#               USERS API (NUEVO)
# ==========================================

# LIST - Obtener todos los usuarios
@app.route("/users", methods=["GET"])
def get_users():
    return jsonify({"users": users})

# GET - Obtener un usuario por ID
@app.route("/users/<int:user_id>", methods=["GET"])
def get_single_user(user_id):
    user = next((u for u in users if u["id"] == user_id), None)
    if user is None:
        return jsonify({"error": "User not found"}), 404
    return jsonify({"user": user})

# CREATE - Añadir un nuevo usuario
@app.route("/users", methods=["POST"])
def add_user():
    global user_id_counter
    data = request.json
    
    # Validar campos mínimos
    if not data or not data.get("name") or not data.get("lastname"):
        return jsonify({"error": "Name and lastname are required"}), 400
        
    user = {
        "id": user_id_counter,
        "name": data["name"],
        "lastname": data["lastname"],
        "address": data.get("address", {}) # Objeto de dirección anidado
    }
    users.append(user)
    user_id_counter += 1
    return jsonify({"message": "User added!", "user": user}), 201

# UPDATE - Actualizar un usuario por ID
@app.route("/users/<int:user_id>", methods=["PUT"])
def update_user(user_id):
    user = next((u for u in users if u["id"] == user_id), None)
    if user is None:
        return jsonify({"error": "User not found"}), 404
        
    data = request.json
    if "name" in data: user["name"] = data["name"]
    if "lastname" in data: user["lastname"] = data["lastname"]
    if "address" in data: user["address"] = data["address"]
        
    return jsonify({"message": "User updated!", "user": user})

# DELETE - Eliminar un usuario por ID
@app.route("/users/<int:user_id>", methods=["DELETE"])
def delete_user(user_id):
    global users
    user = next((u for u in users if u["id"] == user_id), None)
    if user is None:
        return jsonify({"error": "User not found"}), 404
        
    users = [u for u in users if u["id"] != user_id]
    return jsonify({"message": "User deleted!", "user": user})

if __name__ == "__main__":
    # Start development server
    app.run(debug=True)