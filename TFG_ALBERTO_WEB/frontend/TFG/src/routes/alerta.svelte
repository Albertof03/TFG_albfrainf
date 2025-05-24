<script>
  import { onMount, tick } from "svelte";
  import { initMap, clearEarthquakeMarkers, drawTectonicPlates, addEarthquakeMarkers, removeTectonicPlates, loadTectonicPlates } from "../lib/heatmap";

  export let terremotosVisible;
  export let tectonicPlatesVisible;
  let data; 
  let loading = true;
  export let map = null;
  let menuVisible = true;
  let optVisible = false;

  const initializeMap = async () => {
    await tick();
    const mapContainer = document.getElementById("map");
    if (!mapContainer) return;

    if (map !== null) {
        map.remove();
        map = null;
    }

    map = initMap();
    await loadTectonicPlates();
  };

  const toggleTectonicPlates = () => {
    if (tectonicPlatesVisible) {
      removeTectonicPlates(map);
    } else {
      drawTectonicPlates(map);
    }
    tectonicPlatesVisible = !tectonicPlatesVisible;
  };

  onMount(async () => {
      loading = true;

      if (!terremotosVisible) {
          await initializeMap();

          const token = localStorage.getItem("token");
          if (!token) {
              alert("Debes iniciar sesión.");
              loading = false;
              return;
          }

          const payload = JSON.parse(atob(token.split(".")[1]));
          const userId = payload.id;

          try {
              const response = await fetch(`http://localhost:10000/alerta?userId=${userId}`, {
                  method: "GET",
                  headers: { Authorization: `Bearer ${token}` }
              });

              if (!response.ok) throw new Error(`Error HTTP ${response.status}`);

              data = await response.json();
          } catch (error) {
              console.error("🚨 Error obteniendo datos:", error);
          }
      }

      loading = false;
  });
  const toggleMenu = () => {
    menuVisible = !menuVisible;
  };
  const toggleopt = () => {
    optVisible = !optVisible;
  };
</script>

<main>
  {#if !loading}
    <button class="menu2" on:click={toggleopt}>☰</button>
  {/if}
  <div id="map-container">
    <div id="map"></div>
    {#if !loading}
      {#if optVisible}
          <div class="map-controls">
            <button class="close-btn" on:click={() => (optVisible = false)}>✖</button>
            Options:
            <label>
              <input 
                type="checkbox" 
                bind:checked={terremotosVisible}
                on:change={() => {
                  if (terremotosVisible) {
                    addEarthquakeMarkers(map, data);
                  } else {
                    clearEarthquakeMarkers(map);
                  }
                }} 
              />
              Earthquakes that have occurred near your area
            </label>      
            <label>
              <input type="checkbox" on:change={toggleTectonicPlates} checked={tectonicPlatesVisible} />
              Plate Boundaries
            </label>
          </div>
        {/if}
    {/if}
  </div>
  <button class="menu-toggle" on:click={toggleMenu}>⋮⋮⋮</button>
  <div 
    class="menu" 
    style="transform: {menuVisible ? 'translateX(0)' : 'translateX(-100%)'}"
  >
    <button class="close-btn_1" on:click={() => (menuVisible = false)}>✖</button>
    <div class="header-container">
      <h1>
        <i class="fas fa-globe"></i>
        <span class="title-text">Earthquake Monitor</span>
      </h1>
    </div>
    <div class="controls">
      <div id="info-box">
        {#if loading}
          <p class="loading-text">Loading parameters...</p>
        {:else}
          <div class="info-content">
            <div><strong>Average magnitude of a possible earthquake in your area:</strong> {data?.magnitude || "N/A"}</div>
            <div><strong>Possible number of earthquakes in your area:</strong> {data?.numeroter || "N/A"}</div>
          </div>
        {/if}
      </div>
      <button class="btn" on:click={() => window.location.replace("/")}>Volver</button>
    </div>
      
</main>

<style>

  .info-content {
    background: rgba(255, 255, 255, 0.8);
    padding: 15px;
    border-radius: 8px;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    margin-top: 20px;
    font-size: 14px;
    color: #2a8492;
  }

  .info-content div {
    margin-bottom: 10px;
  }

  .info-content strong {
    font-weight: 600;
    color: #2a8492;
  }

  .loading-text {
    color: #ffffff;
    font-size: 16px;
    font-weight: bold;
    text-align: center;
  }

  .menu2 {
    position: absolute;
    top: 90px;
    right: 10px;
    background: #4ca1af;;
    color: white;
    border: none;
    padding: 5px 10px;
    font-size: 20px;
    cursor: pointer;
    border-radius: 5px;
    z-index: 300;
  }

  .menu2:hover {
    background: #3b8a99;
  }

  .map-controls{
    position: absolute;
    top: 20px;
    right: 60px;
    background: rgba(255, 255, 255, 0.9);
    padding: 10px;
    border-radius: 5px;
    box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.3);
    font-size: 14px;
    z-index: 1000;
    color: black;
  }

  .map-controls label {
    display: block;
    cursor: pointer;
    font-size: 14px;
    margin-bottom: 5px;
  }
  .close-btn_1 {
    position: absolute;
    top: 10px;
    right: 10px;
    background: rgb(29, 92, 252);
    color: white;
    border: none;
    padding: 5px 10px;
    font-size: 11px;
    cursor: pointer;
    border-radius: 5px;
  }
  :global(body) {
  margin: 0;
  padding: 0;
  font-family: 'Poppins', sans-serif;
  background: linear-gradient(to bottom, #2c3e50, #4ca1af);
  height: 100vh;
  width: 100vw;
}

  main {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100vh;
    width: 100vw;
    position: absolute;
    top: 0;
    left: 0;
    background: linear-gradient(to bottom, #2c3e50, #4ca1af);
  }
  #map-container {
    position: relative;
    top: 0;
    left: 0;
    width: 100%;
    height:100%;
    display: flex;
    justify-content: flex-end;
    align-items: center;
    z-index: 10;
  }

  #map {
    width:100%;
    height: 100%;
    position: relative;
  }
  .menu-toggle {
    position: absolute;
    top: 20px;
    left: 20px;
    background: #4ca1af;;
    color: white;
    border: none;
    padding: 5px 10px;
    font-size: 20px;
    cursor: pointer;
    border-radius: 5px;
    z-index: 200;
  }

  .header-container {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #4ca1af;
    color: white;
    margin-top: 20px;
  }

  .header-container h1 {
    font-size: 22px;
    font-weight: bold;
    display: flex;
    align-items: center;

  }

  .header-container i {
    font-size: 28px;
  }

  .menu {
    position: fixed;
    top: 0;
    left: 0;
    width: 250px;
    height: 100vh;
    background: #4ca1af;
    padding: 20px;
    transition: transform 0.3s ease-in-out;
    z-index: 200;
  }

  .close-btn {
    position: absolute;
    top: 7px;
    right: 7px;
    background: rgb(29, 92, 252);
    color: white;
    border: none;
    padding: 2px 6px;
    font-size: 13px;
    cursor: pointer;
    border-radius: 5px;
  }

  .controls {
    display: flex;
    flex-direction: column;
    gap: 12px;
    align-items: center;
    margin-top: 20px;
  }

  .btn {
    padding: 12px 24px;
    background: #72aae3;
    color: white;
    font-weight: bold;
    border: none;
    cursor: pointer;
    border-radius: 8px;
    transition: background 0.3s, transform 0.2s;
  }

  .btn:hover {
    background: #5b9dd8;
    transform: scale(1.05);
  }
  .title-text {
  margin-left: 10px;
}


</style>