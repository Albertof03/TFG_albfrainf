import login from '../src/routes/login.svelte';
import perfil from '../src/routes/perfil.svelte';
import register from '../src/routes/register.svelte';

const routes={
    '/login':login,
    '/perfil':perfil,
    '/register':register
}

export default routes