<script>
  import axios from "axios";
  import { push } from "svelte-spa-router";

  export let showRegister;
  export let username;
  export let email;
  export let street;
  export let number;
  export let postal_code;
  export let city;
  export let province;
  export let country;
  export let password;
  export let error;

  async function register() {
    try {
      await axios.post("http://localhost:10000/register", {
        username, email, password, street, number, postal_code, city, province, country
      });
      alert("Registro exitoso, ahora inicia sesión.");
      showRegister = false;
      push('/login');
    } catch (err) {
      error = err.response?.data?.error || "Error en el registro";
    }
  }

  function validateAndRegister() {
    const emailRegex = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;

    if (!emailRegex.test(email)) {
      alert("Por favor, introduce un correo electrónico válido (ejemplo@gmail.com)");
      return;
    }

    if (!/^\d+$/.test(postal_code)) {
      alert("El código postal debe contener solo números");
      return;
    }

    register();
  }
</script>

<div class="login-container">
  <h2>Register</h2>
  {#if error}
    <p class="error-message">{error}</p>
  {/if}

  <div class="login-form">
    <div class="input-field">
      <label>Username:</label>
      <input bind:value={username} placeholder="Username" required/>
    </div>
    <div class="input-field">
      <label>Email:</label>
      <input bind:value={email} placeholder="Email" type="email" />
    </div>
    <div class="input-field">
      <label>Password:</label>
      <input bind:value={password} placeholder="••••••••" type="password" required/>
    </div>
    <div class="input-field">
      <label>Street:</label>
      <input bind:value={street} placeholder="street" />
    </div>
    <div class="input-field">
      <label>Number/Floor:</label>
      <input bind:value={number} placeholder="Number/Floor" />
    </div>
    <div class="input-field">
      <label>Zip code:</label>
      <input bind:value={postal_code} placeholder="Zip code" type="number" required />
    </div>
    <div class="input-field">
      <label>City or Municipality:</label>
      <input bind:value={city} placeholder="City or Municipality" required/>
    </div>
    <div class="input-field">
      <label>Province or State:</label>
      <input bind:value={province} placeholder="Province or State" required/>
    </div>
    <div class="input-field">
      <label>Country:</label>
      <input bind:value={country} placeholder="PaCountryís" required/>
    </div>
  </div>

  <div class="button-group">
    <button class="secondary-btn" on:click={() => { showRegister = false; window.location.replace("/"); }}>Return</button>
    <button class="primary-btn" on:click={validateAndRegister}>Register</button>
  </div>
</div>

<style>
.login-container {
  background: rgba(255, 255, 255, 0.1);
  color: white;
  padding: 25px;
  border-radius: 12px;
  box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.5);
  width: 90%;
  max-width: 800px;
  text-align: center;
}

.login-container h2 {
  font-size: 1.8rem;
  margin-bottom: 15px;
}

.error-message {
  color: #ff4d4d;
  font-weight: bold;
}

.login-form {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 30px;
  margin-bottom: 20px;
}

.input-field label {
  display: block;
  margin-bottom: 5px;
  font-weight: bold;
}

.input-field input {
  width: 100%;
  padding: 10px;
  border: none;
  border-radius: 6px;
  background: rgba(255, 255, 255, 0.2);
  color: white;
  font-size: 1rem;
}

.input-field {
  margin-bottom: 15px;
}

.input-field input::placeholder {
  color: rgba(255, 255, 255, 0.7);
}

.button-group {
  display: flex;
  justify-content: space-between;
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
  background: #3788d9;
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
