<script>
  import { onMount } from "svelte";
  import { user, logout } from "../stores/auth";

  let editando = false;
  let perfilEditado = {};

  $: perfilEditado = { ...$user }; 

  function cerrarSesion() {
    logout();
    window.location.replace("/");
  }

  async function guardarCambios() {
    try {
      const token = localStorage.getItem('token')?.replace(/"/g, '');
      console.log("🔑 Token antes de enviar petición:", token);

      const respuesta = await fetch("http://localhost:10000/perfil", {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify(perfilEditado)
      });

      console.log("📨 Respuesta del servidor:", respuesta);

      if (respuesta.ok) {
        const datosActualizados = await respuesta.json();
        console.log("🆕 Datos actualizados recibidos:", datosActualizados);
        user.set(datosActualizados.user);
        editando = false;
      } else {
        const error = await respuesta.json();
        alert(error.message || "Error al actualizar el perfil");
      }
    } catch (error) {
      console.error("❌ Error de red: ", error);
    }
  }
</script>

{#if $user}
  <div class="perfil-container">
    <h2>Profile</h2>
    {#if editando}
      <div class="profile-form">
        <div class="profile-field">
          <label>Username:</label>
          <input type="text" bind:value={perfilEditado.username} disabled />
        </div>
        <div class="profile-field">
          <label>Email:</label>
          <input type="email" bind:value={perfilEditado.email} disabled />
        </div>
        <div class="profile-field">
          <label>Street:</label>
          <input type="text" bind:value={perfilEditado.street} />
        </div>
        <div class="profile-field">
          <label>Number/Floor:</label>
          <input type="text" bind:value={perfilEditado.number} />
        </div>
        <div class="profile-field">
          <label>Zip code:</label>
          <input type="text" bind:value={perfilEditado.postal_code} />
        </div>
        <div class="profile-field">
          <label>City or Municipality:</label>
          <input type="text" bind:value={perfilEditado.city} />
        </div>
        <div class="profile-field">
          <label>Province or State:</label>
          <input type="text" bind:value={perfilEditado.province} />
        </div>
        <div class="profile-field">
          <label>Country:</label>
          <input type="text" bind:value={perfilEditado.country} />
        </div>
      </div>
    {:else}
      <div class="profile-info">
        <p><strong>Username:</strong> <span class="user-data">{$user.username}</span></p>
        <p><strong>Email:</strong> <span class="user-data">{$user.email}</span></p>
        <p><strong>Street:</strong> <span class="user-data">{$user.street}</span></p>
        <p><strong>Number/Floor:</strong> <span class="user-data">{$user.number}</span></p>
        <p><strong>Zip code:</strong> <span class="user-data">{$user.postal_code}</span></p>
        <p><strong>City or Municipality:</strong> <span class="user-data">{$user.city}</span></p>
        <p><strong>Province or State:</strong> <span class="user-data">{$user.province}</span></p>
        <p><strong>Country:</strong> <span class="user-data">{$user.country}</span></p>
      </div>
    {/if}

    <div class="button-group">
      {#if editando}
        <button class="primary-btn" on:click={guardarCambios}>Save</button>
        <button class="secondary-btn" on:click={() => (editando = false)}>Cancel</button>
      {:else}
        <button class="primary-btn" on:click={() => window.location.replace("/")}>Go to the beginning</button>
        <button class="primary-btn" on:click={() => (editando = true)}>Edit</button>
        <button class="secondary-btn" on:click={cerrarSesion}>logout</button>
      {/if}
    </div>
  </div>
{:else}
  <div class="perfil-container">
    <h1>Profile</h1>
    <p>You are not logged in</p>
    <button class="primary-btn" on:click={() => window.location.replace("/")}>Go to the beginning</button>
  </div>
{/if}

<style>
.perfil-container { 
  background: rgba(255, 255, 255, 0.1);
  color: white;
  padding: 25px;
  border-radius: 12px;
  box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.5);
  width: 90%;
  max-width: 800px;
  text-align: center;
  margin-top: 0px;
}

.perfil-container h2 {
  font-size: 1.8rem;
  margin-bottom: 15px;
}

.profile-info p {
  font-size: 1.2rem;
  margin: 10px 0;
}

.user-data {
  color: #ffcc00;
}

.profile-form {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 20px;
}

.profile-field {
  margin-bottom: 15px;
}

.profile-field label {
  display: block;
  font-weight: bold;
  margin-bottom: 5px;
}

.profile-field input {
  width: 100%;
  padding: 10px;
  border: none;
  border-radius: 6px;
  background: rgba(255, 255, 255, 0.2);
  color: white;
  font-size: 1rem;
}

.profile-field input::placeholder {
  color: rgba(255, 255, 255, 0.7);
}

.button-group {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  margin-top: 20px;
}

.primary-btn, .secondary-btn {
  padding: 10px 15px;
  border: none;
  border-radius: 6px;
  font-size: 1rem;
  font-weight: bold;
  cursor: pointer;
  transition: background 0.3s ease-in-out;
}

.primary-btn {
  background: #217ad3;
  color: white;
}

.primary-btn:hover {
  background: #267acf;
}

.secondary-btn {
  background: #fc4a4a;
  color: white;
}

.secondary-btn:hover {
  background: #d43c3c;
}
</style>
