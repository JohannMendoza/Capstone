function toggle() {
    const currentPage = window.location.pathname.split("/").pop();
    
    if (currentPage === "sign-in.html") {
        window.location.href = "sign-up.html";
    } else {
        window.location.href = "sign-in.html";
    }
}
