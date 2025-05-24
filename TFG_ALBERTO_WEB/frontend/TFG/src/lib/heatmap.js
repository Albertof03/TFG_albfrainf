import L  from 'leaflet';
import 'leaflet.heat';
import Papa from 'papaparse';

let currentHeatLayer = null;
let currentPlatesLayer = null;
let currentInteractiveLayer = null;
let BOUNDING_CSV = [];
let MAGNITUDE_CSV = [];
let EARTHQUAKE_COUNT_CSV = [];
let TECTONIC_PLATES=[];
let earthquakeMarkers = [];
export let minfe=null;
export let maxfe=null;

const getMinMax = (data, column) => {
    let values = data
        .map(row => {
            const dateString = row[column];
            if (dateString && !isNaN(new Date(dateString).getTime())) {
                const date = new Date(dateString);
                return date.getTime();
            }
            return null;
        })
        .filter(time => time !== null);

    if (values.length === 0) {
        return { min: 0, max: 1, minDate: null, maxDate: null };
    }

    let minTimestamp = Math.min(...values);
    let maxTimestamp = Math.max(...values);

    const formatDate = timestamp => new Date(timestamp).toISOString().split("T")[0];

    return {
        minDate: formatDate(minTimestamp),
        maxDate: formatDate(maxTimestamp)
    };
};


const loadCSV = async (url) => {
    const response = await fetch(url);
    const csvText = await response.text();
    return new Promise((resolve) => {
        Papa.parse(csvText, {
            header: true,
            skipEmptyLines: true,
            dynamicTyping: true,
            complete: (results) => resolve(results.data)
        });
    });
};

export const loadCSVData = async () => {
    const boundingData = await loadCSV('/bounding_boxs.csv'); 
    const magnitudeData = await loadCSV('/predicciones_magnitud.csv'); 
    const earthquakeCountData = await loadCSV('/predicciones_terremotos.csv'); 

    MAGNITUDE_CSV = magnitudeData;
    EARTHQUAKE_COUNT_CSV = earthquakeCountData;

    BOUNDING_CSV = boundingData.map(row => {
        let coordenadas = [];
    
        if (typeof row.coordinates === "string" && row.coordinates.trim() !== "") {
            try {
                const cleanedCoordinates = row.coordinates.replace(/^"|"$/g, '').replace(/""/g, '"');
                coordenadas = JSON.parse(cleanedCoordinates);
            } catch (error) {
                console.error("Error al parsear coordenadas JSON:", row.coordinates, error);
            }
        } else {
            console.warn("Coordenadas no definidas o vacías para fila:", row);
        }
    
        return {
            new_cluster_id: row.new_cluster_id,
            coordenadas: coordenadas
        };
    });
    

    const { minDate, maxDate } = getMinMax(MAGNITUDE_CSV, "fecha");
    minfe = minDate;
    maxfe = maxDate;
    return {minfe, maxfe}
};

export const initMap = () => { 
    const map = L.map('map', {
        zoomControl: false
    }).setView([40.5678, -75.1234], 2);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors'
    }).addTo(map);

    L.control.zoom({
        zoomInTitle: 'Acercar',
        zoomOutTitle: 'Alejar',
        position: 'topright'
    }).addTo(map);

    return map;
};

export const createHeatMap = (map, selectedIndex) => {
    if (!map || !BOUNDING_CSV.length || !MAGNITUDE_CSV.length) {
        console.error("No hay datos para generar el heatmap de magnitud.");
        return;
    }

    removeHeatMap(map);

    let heatData = [];
    let interactiveAreas = [];
    const bboxColumns = Object.keys(MAGNITUDE_CSV[0] || {}).filter(col => /^bbox\d+$/.test(col));
    const selectedRow = MAGNITUDE_CSV.find(row => row.fecha === selectedIndex) || {};
    if (Object.keys(selectedRow).length === 0) {
        console.error("No se encontró ninguna fila con la fecha proporcionada.");
        return;
    }
    const intensities = bboxColumns.map(col => selectedRow[col] || 0);

    BOUNDING_CSV.forEach((row, index) => {
        if (!row.coordenadas) return;
        let bbox = Array.isArray(row.coordenadas[0]) ? row.coordenadas.flat() : row.coordenadas;
        let latitudes = bbox.map(point => point[1]);
        let longitudes = bbox.map(point => point[0]);

        if (!latitudes.length || !longitudes.length) return;

        let center_lat = latitudes.reduce((a, b) => a + b, 0) / latitudes.length;
        let center_lon = longitudes.reduce((a, b) => a + b, 0) / longitudes.length;
        let intensity = intensities[index] || 0;

        heatData.push([center_lat, center_lon, intensity]);

        let invisibleArea = L.circleMarker([center_lat, center_lon], {
            radius: 15,
            opacity: 0,
            fillOpacity: 0
        }).on('click', () => {
            L.popup()
                .setLatLng([center_lat, center_lon])
                .setContent(
                    `<b>Latitude:</b> ${center_lat.toFixed(3)}<br>
                     <b>Longitude:</b> ${center_lon.toFixed(3)}<br>
                     <b>Intensity:</b> ${intensity.toFixed(2)}`
                )
                .openOn(map);
        });
        interactiveAreas.push(invisibleArea);
    });

    if (!heatData.length) {
        console.error("No se generaron puntos válidos para el heatmap.");
        return;
    }

    currentHeatLayer = L.heatLayer(heatData, { 
        radius: 25,
        blur: 15,
        maxZoom: 4,
        minOpacity: 0.6,
        gradient: {
            0.1: 'blue',
            0.3: 'cyan',
            0.5: 'lime',
            0.6: 'yellow',
            1.0: 'red'
        }
    }).addTo(map);

    currentInteractiveLayer = L.layerGroup(interactiveAreas).addTo(map);
};


export const createEarthquakeCountHeatMap = (map, selectedIndex) => {
        if (!map || !BOUNDING_CSV.length || !EARTHQUAKE_COUNT_CSV.length) {
            console.error("No hay datos para generar el heatmap de número de terremotos.");
            return;
        }
    
        removeHeatMap(map);
    
        let heatData = [];
        let interactiveAreas = [];
        const bboxColumns = Object.keys(EARTHQUAKE_COUNT_CSV[0] || {}).filter(col => /^bbox\d+$/.test(col));
        const selectedRow = EARTHQUAKE_COUNT_CSV.find(row => row.fecha === selectedIndex) || {};
    
        if (Object.keys(selectedRow).length === 0) {
            console.error("No se encontró ninguna fila con la fecha proporcionada.");
            return;
        }
        const counts = bboxColumns.map(col => selectedRow[col] || 0);
        
        const maxCount = Math.max(...counts, 1);
        
        BOUNDING_CSV.forEach((row, index) => {
            if (!row.coordenadas) return;
            let bbox = Array.isArray(row.coordenadas[0]) ? row.coordenadas.flat() : row.coordenadas;
            let latitudes = bbox.map(point => point[1]);
            let longitudes = bbox.map(point => point[0]);
    
            if (!latitudes.length || !longitudes.length) return;
    
            let center_lat = latitudes.reduce((a, b) => a + b, 0) / latitudes.length;
            let center_lon = longitudes.reduce((a, b) => a + b, 0) / longitudes.length;
            let count = counts[index] || 0;
            
            let normalizedCount = Math.log1p(count) / Math.log1p(maxCount);
            heatData.push([center_lat, center_lon, normalizedCount]);
    
        let invisibleArea = L.circleMarker([center_lat, center_lon], {
                radius: 15,
                opacity: 0,
                fillOpacity: 0
            }).on('click', () => {
                L.popup()
                    .setLatLng([center_lat, center_lon])
                    .setContent(
                        `<b>Latitude:</b> ${center_lat.toFixed(3)}<br>
                        <b>Longitude:</b> ${center_lon.toFixed(3)}<br>
                        <b>Possible Number of Earthquakes:</b> ${Math.ceil(count)}`
                    )
                    .openOn(map);
            });
            interactiveAreas.push(invisibleArea);
        });
    
        if (!heatData.length) {
            console.error("No se generaron puntos válidos para el heatmap de número de terremotos.");
            return;
        }
    
        currentHeatLayer = L.heatLayer(heatData, { 
            radius: 25,
            blur: 15,
            maxZoom: 4,
            minOpacity: 0.6,
            gradient: {
                0.0: '#2a2a72',
                0.1: '#009FFD', 
                0.3: '#00FFA2', 
                0.5: '#FFD500',  
                0.8: '#FF6A00', 
                1.0: '#D00000' 
            }
            
        }).addTo(map);
    
        currentInteractiveLayer = L.layerGroup(interactiveAreas).addTo(map);
};


export const removeHeatMap = (map) => {
    if (currentHeatLayer) {
        map.removeLayer(currentHeatLayer);
        currentHeatLayer = null;
    }
};


export const loadTectonicPlates = async () => {
    const platesData = await loadCSV('/placas_tectonicas.csv');

    const platesMap = {};
    platesData.forEach(row => {
        const plate = row.plate;
        const lat = parseFloat(row.lat);
        const lon = parseFloat(row.lon);

        if (!platesMap[plate]) {
            platesMap[plate] = [];
        }
        platesMap[plate].push([lat, lon]);
    });

    TECTONIC_PLATES = Object.values(platesMap);
};

export const drawTectonicPlates = (map) => {
    if (!map || !TECTONIC_PLATES.length) {
        console.error("No hay datos de placas tectónicas.");
        return;
    }

    removeTectonicPlates(map);

    const MAX_LONGITUDE_JUMP = 180;
    const segmentedPlates = TECTONIC_PLATES.map(coords => {
        let segments = [];
        let currentSegment = [coords[0]];

        for (let i = 1; i < coords.length; i++) {
            const [prevLat, prevLon] = coords[i - 1];
            const [currLat, currLon] = coords[i];

            if (Math.abs(currLon - prevLon) > MAX_LONGITUDE_JUMP) {
                segments.push(currentSegment);
                currentSegment = [];
            }

            currentSegment.push([currLat, currLon]);
        }

        if (currentSegment.length > 1) {
            segments.push(currentSegment);
        }

        return segments;
    }).flat();

    currentPlatesLayer = L.layerGroup(
        segmentedPlates.map(segment => 
            L.polyline(segment, { color: '#32CD32', weight: 2 })
        )
    ).addTo(map);
};

export const removeTectonicPlates = (map) => {
    if (currentPlatesLayer) {
        map.removeLayer(currentPlatesLayer);
        currentPlatesLayer = null;
    }
};

export const clearEarthquakeMarkers = (map) => {
    earthquakeMarkers.forEach(marker => map.removeLayer(marker));
    earthquakeMarkers = [];
};

export const addEarthquakeMarkers = (map, earthquakes) => {
    clearEarthquakeMarkers(map);
    
    if (earthquakes && Array.isArray(earthquakes.terremotos)) {
        earthquakes.terremotos.forEach(eq => {
            const marker = L.circleMarker([eq.latitud, eq.longitud], {
                radius: 8,
                fillColor: 'red',
                color: '#900',
                weight: 1,
                opacity: 1,
                fillOpacity: 0.8
            }).addTo(map)
              .bindPopup(
                `<b>Id:</b> ${eq.id}<br>
                 <b>Title:</b> ${eq.titulo}<br>
                 <b>Date:</b> ${eq.fecha}<br>
                 <b>Magnitude:</b> ${eq.magnitud}<br>
                 <b>Distance:</b> ${eq.distancia} km<br>`
              );
            
            earthquakeMarkers.push(marker);
        });
    } else {
        console.error("No se encontró la propiedad 'terremotos' en los datos de terremotos.");
    }
};