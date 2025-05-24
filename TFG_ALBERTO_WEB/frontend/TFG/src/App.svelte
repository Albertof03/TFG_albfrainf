<script>
  import { onMount } from "svelte";
  import { loadUser } from "../src/stores/auth";
  import { loadCSVData, initMap, createHeatMap, createEarthquakeCountHeatMap, loadTectonicPlates, drawTectonicPlates, removeHeatMap, removeTectonicPlates } from "./lib/heatmap.js";
  import { location } from "svelte-spa-router";
  import Menu from "./routes/menu.svelte";
  import Perfil from "./routes/perfil.svelte";
  import Register from "./routes/register.svelte";
  import Login from "./routes/login.svelte";
  import Alerta from "../src/routes/alerta.svelte";

  let map = null;
  let selectedDate = "2025-01-21";
  let heatmapVisible = false;
  let earthquakeCountVisible = false;
  let tectonicPlatesVisible = false;
  let terremotosVisible = false;
  let menuVisible = true;
  let optVisible = false;

  let showRegister = false;
  let showLogin = false;

  let username = "";
  let email = "";
  let password = "";
  let street = "";
  let number = "";
  let postal_code = "";
  let city = "";
  let province = "";
  let country = "";
  let usernameOrEmail = "";
  let error = "";
  let minfe = null;
  let maxfe = null;

  const initializeMap = async () => {
    if (!map) {
      map = initMap();  
      await loadTectonicPlates();
      const { minfe: minDate, maxfe: maxDate } = await loadCSVData();
      minfe = minDate;
      maxfe = maxDate;
    } else {
      setTimeout(() => {
        map.invalidateSize();
      }, 500);
    }
  };

  onMount(async () => {
    await initializeMap();
    const currentToken = localStorage.getItem("token")?.replace(/^"(.*)"$/, "$1");
    if (currentToken) {
      loadUser();
    }
  });

  const toggleMenu = () => {
    menuVisible = !menuVisible;
  };

  const toggleopt = () => {
    optVisible = !optVisible;
  };

  const toggleHeatmap = async (event) => {
    if (!selectedDate) {
      heatmapVisible = false;
      event.target.checked = false;
      alert("No se ha especificado fecha");
      return;
    }
    
    if (heatmapVisible) {
      removeHeatMap(map);
      heatmapVisible = false;
    } else {
      createHeatMap(map, selectedDate);
      heatmapVisible = true;
      earthquakeCountVisible = false;
    }
  };

  const toggleEarthquakeCountHeatmap = async (event) => {
    if (!selectedDate) {
      earthquakeCountVisible = false;
      event.target.checked = false;
      alert("No se ha especificado fecha");
      return;
    }

    if (earthquakeCountVisible) {
      removeHeatMap(map);
      earthquakeCountVisible = false;
    } else {
      createEarthquakeCountHeatMap(map, selectedDate);
      earthquakeCountVisible = true;
      heatmapVisible = false;
    }
  };

  const toggleTectonicPlates = () => {
    if (tectonicPlatesVisible) {
      removeTectonicPlates(map);
    } else {
      drawTectonicPlates(map);
    }
    tectonicPlatesVisible = !tectonicPlatesVisible;
  };

  const actualizarMapa = (fecha) => {
    selectedDate = fecha;

    if (heatmapVisible) {
      removeHeatMap(map);
      createHeatMap(map, selectedDate);
    }
    if (earthquakeCountVisible) {
      removeHeatMap(map);
      createEarthquakeCountHeatMap(map, selectedDate);
    }
  };

</script>

<main>
  {#if $location != "/perfil" && $location != "/alerta" }
    <button class="menu-toggle" on:click={toggleMenu}>⋮⋮⋮</button>
    <Menu 
      bind:selectedDate
      bind:maxfe
      bind:minfe
      bind:menuVisible
      on:dateChange={(event) => actualizarMapa(event.detail.date)} 
    />
  {/if}

  {#if $location == "/perfil" }
    <Perfil />
  {:else if $location == "/register" }
    <Register
      bind:showRegister
      bind:username
      bind:email
      bind:password
      bind:country
      bind:province
      bind:postal_code
      bind:city
      bind:number
      bind:street
      bind:error
    />
  {:else if $location == "/login" }
    <Login
      bind:usernameOrEmail
      bind:password
      bind:showLogin
      bind:error
    />
  {:else if $location == "/alerta" }
    <Alerta
      bind:terremotosVisible
      bind:tectonicPlatesVisible
    />
  {:else}
  <button class="menu2" on:click={toggleopt}>☰</button>
    <div id="map-container">
      <div id="map"></div>
      {#if optVisible}
        <div class="map-controls">
          <button class="close-btn" on:click={() => (optVisible = false)}>✖</button>
          <div class="heatmap-container">
            Heatmaps:
            <label>
              <input type="checkbox" on:change={toggleHeatmap} bind:checked={heatmapVisible} disabled={earthquakeCountVisible} />
              Average magnitude
            </label>
            
            <label>
              <input type="checkbox" on:change={toggleEarthquakeCountHeatmap} bind:checked={earthquakeCountVisible} disabled={heatmapVisible} />
              Number of Earthquakes
            </label>
          </div>
          <div class="opts">
            Options:
          <label>
            <input type="checkbox" on:change={toggleTectonicPlates} checked={tectonicPlatesVisible} />
            Plate Boundaries
          </label>
        </div>
        </div>
      {/if}
    </div>
  {/if}
</main>


<style>
  main {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100vh;
  width: 100vw;
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
    z-index: 200;
  }

  .menu2:hover {
    background: #3b8a99;
  }

  .map-controls {
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
  .close-btn {
    position: absolute;
    top: 10px;
    right: 10px;
    background: rgb(29, 92, 252);
    color: white;
    border: none;
    padding: 2px 6px;
    font-size: 13px;
    cursor: pointer;
    border-radius: 5px;
  }
</style>
