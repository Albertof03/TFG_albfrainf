import express from "express";
import bodyParser from "body-parser";
import cors from "cors";
import path, { dirname } from "path";
import { fileURLToPath } from "url";
import dotenv from "dotenv";
import bcrypt from "bcryptjs";
import jwt from "jsonwebtoken";
import pg from "pg";
import fetch from "node-fetch";
import fs from "fs";
import Papa from "papaparse";

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
app.use(cors({
  origin: "*", // Puedes restringirlo a tu frontend si es necesario
  methods: ["GET", "POST", "PUT", "DELETE"],
  allowedHeaders: ["Content-Type", "Authorization"]
}));
app.use(bodyParser.json());

const PORT = 3000;
//process.env.PORT || 
// Configuración de PostgreSQL
const pool = new pg.Pool({
  user: process.env.DB_USER,
  host: process.env.DB_HOST,
  database: process.env.DB_NAME,
  password: process.env.DB_PASSWORD,
  port: process.env.DB_PORT,
  ssl: { rejectUnauthorized: false } // 🚨 IMPORTANTE en Koyeb
});

// Clave secreta para JWT
const SECRET_KEY = process.env.JWT_SECRET || "secreto_super_seguro";

// RUTA: Registro de usuarios
app.post("/register", async (req, res) => {
  const {username, email, password, street, number, postal_code, city, province, country} = req.body;
  const hashedPassword = await bcrypt.hash(password, 10);

  try {
    const result = await pool.query(
      "INSERT INTO users (username, email, password, street, number, postal_code, city, province, country) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9) RETURNING id, username, email, street, number, postal_code, city, province, country",
      [username, email, hashedPassword, street, number, postal_code, city, province, country]
    );
    res.json({ user: result.rows[0] });
  } catch (err) {
    res.status(400).json({ error: "El usuario o email ya existen" });
  }
});

// RUTA: Login de usuarios
app.post("/login", async (req, res) => {
  const { usernameOrEmail, password} = req.body;

  const result = await pool.query(
    "SELECT * FROM users WHERE email = $1 OR username = $1",
    [usernameOrEmail]
  );
  if (result.rows.length === 0) return res.status(400).json({ error: "Usuario no encontrado" });

  const user = result.rows[0];
  const validPassword = await bcrypt.compare(password, user.password);
  if (!validPassword) return res.status(400).json({ error: "Contraseña incorrecta" });

  const token = jwt.sign({ id: user.id, username: user.username, email: user.email}, SECRET_KEY, { expiresIn: "1d" });
  res.json({ token });
});


app.get("/perfil", async (req, res) => {
  const authHeader = req.headers.authorization?.trim();
  if (!authHeader || !authHeader.startsWith("Bearer ")) {
    return res.status(401).json({ error: "Acceso denegado, token faltante o mal formado" });
  }

  const token = authHeader.split(" ")[1];
  if (!token) {
    return res.status(401).json({ error: "Token mal formado" });
  }

  try {
    const decoded = jwt.verify(token, SECRET_KEY);
    
    // Obtener los datos del usuario desde la base de datos
    const userResult = await pool.query(
      "SELECT username, email, street, number, postal_code, city, province, country FROM users WHERE id = $1",
      [decoded.id]
    );

    // Si el usuario no existe
    if (userResult.rows.length === 0) {
      return res.status(404).json({ error: "Usuario no encontrado" });
    }

    // Enviar los datos del usuario, incluyendo la dirección
    const user = userResult.rows[0];
    res.json({ user });

  } catch (err) {
    console.error("Error verificando token:", err.message);
    res.status(401).json({ error: "Token inválido o expirado" });
  }
});
app.put("/perfil", async (req, res) => {
  const authHeader = req.headers.authorization?.trim();

  if (!authHeader || !authHeader.startsWith("Bearer ")) {
    console.error("❌ Token faltante o mal formado");
    return res.status(401).json({ error: "Acceso denegado, token faltante o mal formado" });
  }

  const token = authHeader.split(" ")[1];

  if (!token) {
    console.error("❌ Token mal formado después de extracción");
    return res.status(401).json({ error: "Token mal formado" });
  }

  try {
    const decoded = jwt.verify(token, SECRET_KEY);

    const { street, number, postal_code, city, province, country } = req.body;

    const updateQuery = `
      UPDATE users
      SET street = $1, number = $2, postal_code = $3, city = $4, province = $5, country = $6
      WHERE id = $7
      RETURNING username, email, street, number, postal_code, city, province, country
    `;

    const result = await pool.query(updateQuery, [
      street, number, postal_code, city, province, country, decoded.id
    ]);

    if (result.rows.length === 0) {
      console.error("❌ Usuario no encontrado en la base de datos");
      return res.status(404).json({ error: "Usuario no encontrado" });
    }

    res.json({ user: result.rows[0] });
  } catch (err) {
    console.error("🔥 Error actualizando perfil:", err.message);
    res.status(500).json({ error: "Error interno del servidor" });
  }
});


const haversineDistance = (lat1, lon1, lat2, lon2) => {
    const R = 6371; // Radio de la Tierra en km
    const toRad = (angle) => (angle * Math.PI) / 180;
    const dLat = toRad(lat2 - lat1);
    const dLon = toRad(lon2 - lon1);
    const a =
      Math.sin(dLat / 2) * Math.sin(dLat / 2) +
      Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * 
      Math.sin(dLon / 2) * Math.sin(dLon / 2);
    return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
};

app.get("/alerta", async (req, res) => {
  const userId = req.query.userId;

  try {
    const userResult = await pool.query(`
      SELECT street, number, postal_code, city, province, country 
      FROM users 
      WHERE id = $1
    `, [userId]);

    if (userResult.rows.length === 0) {
      console.error("❌ Usuario no encontrado en la BD");
      return res.status(404).json({ error: "Usuario no encontrado" });
    }
    // Obtener los datos de la dirección desde la consulta
    const { street, number, postal_code, city, province, country } = userResult.rows[0];

    // Realiza la consulta a la API de geocodificación usando la búsqueda estructurada
    const geocodeResponse = await fetch(`https://nominatim.openstreetmap.org/search?format=json&postalcode=${encodeURIComponent(postal_code)}&city=${encodeURIComponent(city)}&state=${encodeURIComponent(province)}&country=${encodeURIComponent(country)}`);

    // Obtiene los datos de geocodificación
    const geocodeData = await geocodeResponse.json();


    if (geocodeData.length === 0) {
      console.error("❌ Dirección no encontrada en OpenStreetMap");
      return res.status(404).json({ error: "Dirección no encontrada" });
    }

    const { lat, lon } = geocodeData[0];

    const earthquakesQuery = `
      SELECT id, titulo, fecha, magnitud, ST_Y(coordenadas2d) AS lat, ST_X(coordenadas2d) AS lon
      FROM earthquakes
      ORDER BY ST_DistanceSphere(
        ST_MakePoint(ST_X(coordenadas2d), ST_Y(coordenadas2d)),
        ST_MakePoint($1, $2)
      )
      LIMIT 100;
    `;

    const earthquakesResult = await pool.query(earthquakesQuery, [lon, lat]);
    const terremotosCercanos = earthquakesResult.rows
      .map(eq => ({
        id: eq.id,
        titulo: eq.titulo,
        fecha: eq.fecha,
        magnitud: eq.magnitud,
        latitud: eq.lat,
        longitud:eq.lon,
        distancia: haversineDistance(lat, lon, eq.lat, eq.lon).toFixed(2),
      }))
      .sort((a, b) => new Date(b.fecha) - new Date(a.fecha)); // Ordenamos por fecha descendente

    const csvData = fs.readFileSync("./datos/bounding_boxs.csv", "utf8");
    const parsedBboxes = Papa.parse(csvData, { header: true, dynamicTyping: true }).data;

    let bbox = null;

    try {
      bbox = parsedBboxes.find(bbox => {
        if (!bbox.coordinates || typeof bbox.coordinates !== "string") {
          console.error("❌ bbox.coordinates inválido en:", bbox);
          return false;
        }

        const coords = JSON.parse(bbox.coordinates);
        if (!Array.isArray(coords) || coords.length === 0 || !Array.isArray(coords[0])) {
          console.error("❌ bbox.coordinates mal formateado:", bbox.coordinates);
          return false;
        }

        const bboxArray = coords[0]; // Extraer el primer array interno
        if (bboxArray.length < 4) {
          console.error("❌ bboxArray no tiene suficientes puntos:", bboxArray);
          return false;
        }

        const [minLon, minLat] = bboxArray[0]; // Esquina inferior izquierda
        const [maxLon, maxLat] = bboxArray[2]; // Esquina superior derecha

        return minLat <= lat && lat <= maxLat && minLon <= lon && lon <= maxLon;
      });
    } catch (error) {
      console.error("❌ Error parseando bounding box:", error.message);
    }

    if (!bbox || bbox.new_cluster_id === null) {
      console.error("❌ No se encontró un bbox para estas coordenadas o new_cluster_id es null");
      return res.json({ magnitude: null, terremotos: terremotosCercanos });
    }

    const magData = fs.readFileSync("./datos/predicciones_magnitud.csv", "utf8");
    const numData = fs.readFileSync("./datos/predicciones_terremotos.csv", "utf8");
    const parsedMag = Papa.parse(magData, { header: true, dynamicTyping: true }).data;
    const parsedNum = Papa.parse(numData, { header: true, dynamicTyping: true }).data;

    const filteredMAGData = parsedMag.map(row => {
      return Object.fromEntries(
        Object.entries(row).filter(([key]) => /^bbox\d+$/.test(key))
      );
    });
    const filteredNUMData = parsedNum.map(row => {
      return Object.fromEntries(
        Object.entries(row).filter(([key]) => /^bbox\d+$/.test(key))
      );
    });

    // Construimos el nombre de la columna que queremos buscar
    const targetColumn = `bbox${bbox.new_cluster_id}`;

    // Obtenemos el valor de la primera fila en esa columna
    const magValue = filteredMAGData.length > 0 ? filteredMAGData[0][targetColumn] : null;
    const numValue = filteredNUMData.length > 0 ? filteredNUMData[0][targetColumn] : null;

    return res.json({ magnitude: magValue.toFixed(2), numeroter:Math.ceil(numValue), terremotos: terremotosCercanos });

  } catch (error) {
    console.error("🔥 Error en /alerta:", error.message);
    res.status(500).json({ error: error.message });
  }
});


// Servir frontend
app.use("/", express.static(path.join(__dirname, "../frontend/TFG/dist")));

app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
});
