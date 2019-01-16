/* Copyright 2016 Ramakrishnan Kannan */

void testSmall() {
  fmat A, W, H;
  A.load("/scratch/ramki/datasets/xdata/bpp/A_rnd.csv");
  BPPNMF<fmat> bppnmf2(A, 5);
  INFO << "completed constructor" << endl;
  double t1 = omp_get_wtime();
  bppnmf2.computeNMF();
  double t2 = omp_get_wtime();
  OUTPUT << bppnmf2.getLeftLowRankFactor() << endl;
  OUTPUT << bppnmf2.getRightLowRankFactor() << endl;
  INFO << "Computed Fast NMF with multiple RHS!!!:"
       << norm((A - bppnmf2.getLeftLowRankFactor() *
                        bppnmf2.getRightLowRankFactor().t()),
               "fro")
       << endl;
  INFO << "time taken:" << (t2 - t1) << endl;
}

void testLowRank() {
  fmat A, W, H;
  W.load("/scratch/ramki/datasets/xdata/bpp/w_lr_init.csv");
  H.load("/scratch/ramki/datasets/xdata/bpp/h_lr_init.csv");
  W = W.rows(0, 50000);
  H = H.rows(0, 100000);
  INFO << "compled loading the matrices W.m=" << W.n_rows << " W.n=" << W.n_cols
       << " H.m=" << H.n_rows << " H.n=" << H.n_cols << endl;
  A = W * H.t();
  INFO << "created input matrix m=" << A.n_rows << " n=" << A.n_cols << endl;
  BPPNMF<fmat> bppnmf2(A, 100);
  INFO << "completed constructor" << endl;
  double t1 = omp_get_wtime();
  bppnmf2.computeNMF();
  double t2 = omp_get_wtime();
  INFO << "Computed Fast NMF with multiple RHS!!!:"
       << norm((A - bppnmf2.getLeftLowRankFactor() *
                        bppnmf2.getRightLowRankFactor().t()),
               "fro")
       << endl;
  INFO << "time taken:" << (t2 - t1) << endl;
  bppnmf2.getLeftLowRankFactor().save(
      "/scratch/ramki/datasets/xdata/bpp/w_lr_bpp_output.csv", raw_ascii);
  bppnmf2.getRightLowRankFactor().save(
      "/scratch/ramki/datasets/xdata/bpp/h_lr_bpp_output.csv", raw_ascii);
}
