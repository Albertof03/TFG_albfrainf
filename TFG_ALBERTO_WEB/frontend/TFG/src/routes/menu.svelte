<script>
  import { createEventDispatcher, onMount } from "svelte";
  import { push, location } from "svelte-spa-router";
  import { user, logout } from "../stores/auth";

  export let selectedDate;
  export let minfe;
  export let maxfe;
  export let menuVisible;
  let errorMessage = "";

  const formatDate = (timestamp) => {
    return timestamp ? new Date(timestamp).toISOString().split("T")[0] : undefined;
  };

  $: minDate = formatDate(minfe);
  $: maxDate = formatDate(maxfe);

const dispatch = createEventDispatcher();

$: minTimestamp = minDate ? new Date(minDate).getTime() : undefined;
$: maxTimestamp = maxDate ? new Date(maxDate).getTime() : undefined;
$: selectedTimestamp = selectedDate ? new Date(selectedDate).getTime() : undefined;


const updateDate = (event) => {
    selectedTimestamp = Number(event.target.value);
    selectedDate = new Date(selectedTimestamp).toISOString().split("T")[0];
    dispatch("dateChange", { date: selectedDate });
  };



  $: {
    if (selectedDate && minDate && maxDate) {
      if (selectedDate < minDate || selectedDate > maxDate) {
        errorMessage = "No hay datos disponibles para esta fecha.";
      } else {
        errorMessage = "";
      }
    }
  }
</script>

<main>
  <div 
    class="menu" 
    style="transform: {menuVisible ? 'translateX(0)' : 'translateX(-100%)'}"
  >
    <button class="close-btn" on:click={() => (menuVisible = false)}>✖</button>
    <div class="header-container">
      <h1>
        <i class="fas fa-globe"></i>
        <span class="title-text">Earthquake Monitor</span>
      </h1>
    </div>  
    <div class="controls">
      {#if $location == "/" }
        {#if $user}
          <div class="button-container">
            <button class="btn" on:click={() => push('/alerta')}>Alert</button>
            <button class="btn" on:click={() => push('/perfil')}>Profile</button>
          </div>
          <button class="btn logout" on:click={logout}>logout ({$user.username})</button>
        {#if errorMessage}
          <p class="error-message">{errorMessage}</p>
        {/if}
        {:else}
          <div class="button-container">
            <button class="btn" on:click={() => push('/register')}>Register</button>
            <button class="btn" on:click={() => push('/login')}>Login</button>
          </div>
        {/if}
        <input 
        type="date" 
        bind:value={selectedDate} 
        class="date-picker" 
        min={minDate} 
        max={maxDate} 
        >
        <div class="slider-container">
          <input 
            type="range" 
            min={minTimestamp} 
            max={maxTimestamp} 
            step={86400000}  
            bind:value={selectedTimestamp} 
            on:input={updateDate}
            class="wide-slider"
          >
        </div>
      {:else}
        {#if $location == "/login"}
          <button class="btn" on:click={() => push('/register')}>Register</button>
        {:else if $location == "/register"}
          <button class="btn" on:click={() => push('/login')}>Login</button>
        {:else if $location == "/alerta"}
          <button class="btn" on:click={() => window.location.replace("/")}>Return</button>
        {/if}
      {/if}
    </div>
  </div>
</main>
<style>
  .slider-container {
      display: flex;
      flex-direction: column;
      align-items: center;
      margin-top: 10px;
      width: 100%;
    }

    input[type="range"] {
      width: 90%;
      -webkit-appearance: none;
      height: 8px;
      background: #72aae3;
      border-radius: 5px;
      outline: none;
      transition: background 0.3s;
    }

    input[type="range"]::-webkit-slider-thumb {
      -webkit-appearance: none;
      width: 18px;
      height: 18px;
      background: white;
      border: 2px solid #72aae3;
      border-radius: 50%;
      cursor: pointer;
    }

    input[type="range"]:hover {
      background: #5b9dd8;
    }

   .date-picker {
    font-size: 16px;
    cursor: pointer;
    text-align: center;
    color: white;
    border: none;
    padding: 10px;
    border-radius: 8px;
    transition: background 0.3s, transform 0.2s;
  }

  .date-picker:hover {
    transform: scale(1.05);
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
    top: 10px;
    right: 10px;
    background: rgb(29, 92, 252);
    color: white;
    border: none;
    padding: 5px 10px;
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

  .logout {
    background: #d32f2f;
  }

  .logout:hover {
    background: #b71c1c;
  }

  .dropdown, .date-picker {
    padding: 10px;
    border-radius: 8px;
    font-weight: bold;
  }

  .date-picker {
    border: 1px solid #ccc;
    font-size: 16px;
    background: white;
    color: #333;
  }

  .error-message {
    color: red;
    font-weight: bold;
    margin-top: 5px;
  }
  .title-text {
  margin-left: 10px;
}
.button-container {
  display: flex;
  gap: 10px;
  justify-content: center;
}
</style>


