<script>
  import axios from "axios";
  import { token, loadUser } from "../stores/auth";

  export let usernameOrEmail;
  export let password;
  export let error;
  export let showLogin;

  async function login() {
      try {
          const res = await axios.post("http://localhost:10000/login", {
              usernameOrEmail,
              password
          });

          const receivedToken = res.data.token;
          localStorage.setItem("token", receivedToken);
          token.set(receivedToken);

          console.log("Token guardado en localStorage:", receivedToken);

          await new Promise((resolve) => setTimeout(resolve, 100));
          loadUser();

          alert("Inicio de sesión exitoso");
          showLogin = false;
          window.location.replace("/")
      } catch (err) {
          error = err.response?.data?.error || "Error en el login";
      }
  }
</script>

<div class="login-container">
  <h2>Login</h2>
  {#if error}
    <p class="error-message">{error}</p>
  {/if}
  
  <div class="login-form">
      <div class="input-field">
          <label>Username or Email:</label>
          <input bind:value={usernameOrEmail} placeholder="Ej: user@gmail.com" />
      </div>
      <div class="input-field">
          <label>Password:</label>
          <input bind:value={password} placeholder="••••••••" type="password" />
      </div>
  </div>

  <div class="button-group">
      <button class="secondary-btn" on:click={() => {showLogin = false; window.location.replace("/");}}>Return</button>
      <button class="primary-btn" on:click={login}>Login</button>
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
  max-width: 400px;
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
  display: flex;
  flex-direction: column;
  gap: 15px;
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

.input-field input::placeholder {
  color: rgba(255, 255, 255, 0.7);
}

.button-group {
  display: flex;
  justify-content: space-between;
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
