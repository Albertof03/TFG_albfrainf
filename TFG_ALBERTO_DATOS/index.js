import express from 'express';
import axios from 'axios';
import pkg from 'pg';
const { Client } = pkg;
import cron from 'node-cron';
import cors from 'cors';
import bodyParser from 'body-parser';

const app = express();
app.use(cors());
app.use(bodyParser.json());

const PORT = process.env.PORT || 10001;
const client = new Client({
  user: 'postgres',
  host: 'localhost',
  database: 'MYtfg',
  password: 'r1u2i3z4',
  port: 5432,
});

await client.connect();

const API_URL = 'https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson';

async function fetchEarthquakeData() {
  try {
    const endDate = new Date();
    const startDate = new Date();
    startDate.setDate(endDate.getDate() - 10);

    const formatDate = (date) => {
      const year = date.getFullYear();
      const month = String(date.getMonth() + 1).padStart(2, '0');
      const day = String(date.getDate()).padStart(2, '0');
      return `${year}-${month}-${day}`;
    };

    const starttime = formatDate(startDate);
    const endtime = formatDate(endDate);

    const queryUrl = `${API_URL}&starttime=${starttime}&endtime=${endtime}`;
    //const queryUrl2 = 'https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=2025-04-01&endtime=2025-05-01';
    const response = await axios.get(queryUrl);
    return response.data.features;
  } catch (error) {
    console.error('Error fetching earthquake data:', error);
    return [];
  }
}

async function saveEarthquakeData(data) {
  try {
    const checkQuery = 'SELECT id FROM earthquakes WHERE id = $1';
    const checkResult = await client.query(checkQuery, [data.id]);

    if (checkResult.rows.length > 0) {
      return;
    }

    const query = `
      INSERT INTO earthquakes (
        id, titulo, fecha, magnitud, tipomagnitud, alerta, tsunami,
        distanciaminepicentro, coberturaangular, significancia, coordenadas, coordenadas2d
      ) VALUES (
        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, ST_SetSRID(ST_MakePoint($11, $12, $13), 4326), ST_SetSRID(ST_MakePoint($11, $12), 4326)
      )
    `;
    const values = [
      data.id,
      data.properties.title,
      new Date(data.properties.time).toISOString(),
      data.properties.mag,
      data.properties.magType || null,
      data.properties.alerta || null,
      data.properties.tsunami,
      data.properties.dmin,
      data.properties.gap,
      data.properties.sig,
      data.geometry.coordinates[0], // Longitud (X)
      data.geometry.coordinates[1], // Latitud (Y)
      data.geometry.coordinates[2]  // Altura (Z)
    ];

    await client.query(query, values);
    console.log(`Data stored successfully for ID: ${data.id}`);
  } catch (err) {
    console.error('Error storing data', err);
  }
}

cron.schedule('*/10 * * * *', async () => {
  console.log('Fetching earthquake data...');
  const earthquakeData = await fetchEarthquakeData();

  if (earthquakeData.length > 0) {
    console.log(`Found ${earthquakeData.length} earthquakes.`);

    const promises = earthquakeData.map(async (data) => {
      await saveEarthquakeData(data);
    });

    await Promise.all(promises);

    console.log('All earthquake data has been processed.');
  } else {
    console.log('No earthquake data found.');
  }
});


(async () => {
  console.log('Fetching earthquake data...');
  const earthquakeData = await fetchEarthquakeData();

  if (earthquakeData.length > 0) {
    console.log(`Found ${earthquakeData.length} earthquakes.`);

    const promises = earthquakeData.map(async (data) => {
      await saveEarthquakeData(data);
    });

    await Promise.all(promises);

    console.log('All earthquake data has been processed.');

    console.log('Shutting down the server...');
    process.exit(0);
  } else {
    console.log('No earthquake data found.');
    console.log('Shutting down the server...');
    process.exit(0);
  }
})();

app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
});