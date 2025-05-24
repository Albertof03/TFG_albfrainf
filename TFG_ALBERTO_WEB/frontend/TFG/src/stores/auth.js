import { writable } from "svelte/store";
import { persisted } from "svelte-local-storage-store";
import { get } from "svelte/store";

export const token = persisted("token", "", { serializer: JSON });
export const user = writable(null);

export function loadUser() {
  if (typeof window === "undefined") return;

  let currentToken = get(token);
  if (!currentToken || currentToken === '""' || currentToken.trim() === "") {
    return;
  }
  currentToken = currentToken.replace(/^"|"$/g, "");
  fetch("http://localhost:10000/perfil", {
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${currentToken}`,
    },
  })
    .then((res) => {
      if (!res.ok) throw new Error("Token inválido");
      return res.json();
    })
    .then((data) => user.set(data.user))
    .catch((err) => {
      console.error("Error cargando usuario:", err);
      logout();
    });
}


// Cerrar sesión
export function logout() {
  localStorage.removeItem("token"); 
  token.set("");
  user.set(null);
  window.location.replace("/");
}

